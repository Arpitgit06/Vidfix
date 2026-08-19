"""
video_engine.py – AV-SynthRestore 3D
4K video upscaling using Real-ESRGAN (or PyTorch bicubic fallback) with
optimized GPU pipeline: producer-consumer I/O, dynamic tile calibration,
torch.compile acceleration, and parallel PNG writes.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional Heavy Imports (runtime-detected) ──────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("PyTorch %s detected.", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available; will use OpenCV bicubic upscaling.")

if TORCH_AVAILABLE and hasattr(torch, "cuda") and hasattr(torch.cuda, "OutOfMemoryError"):
    CUDA_OOM_EXCEPTION = torch.cuda.OutOfMemoryError
else:
    class CUDA_OOM_EXCEPTION(Exception):
        pass

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    logger.info("Real-ESRGAN detected.")
except ImportError:
    REALESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN not installed; bicubic fallback active.")


# ── VideoEngine ────────────────────────────────────────────────────────────────

class VideoEngine:
    """
    Orchestrates per-frame AI upscaling to 4K.

    Processing Pipeline (Optimized)
    --------------------------------
    1. Auto-calibrate tile size: try tile=0 (no tiling) first for maximum
       throughput, fall back to configured tile size on OOM.
    2. Apply torch.compile to the neural network for kernel fusion.
    3. Producer-Consumer I/O pipeline:
       - Reader thread:  pre-loads frames from disk into a RAM queue.
       - GPU thread:     pulls from read queue, runs AI upscaling, pushes
                         results to write pool.
       - Writer pool:    multiple threads compress and write PNGs in parallel.
    4. Per-frame OOM fallback to PyTorch bicubic / OpenCV Lanczos.
    5. Async telemetry updates every 2 seconds while pipeline runs.
    """

    TARGET_W: int = 3840
    TARGET_H: int = 2160

    def __init__(
        self,
        config:   Optional[dict]     = None,
        telemetry_callback: Optional[Callable] = None,
    ) -> None:
        cfg = config or {}
        self.batch_size    = int(cfg.get("batch_size",  4))
        self.tile_size     = int(cfg.get("tile_size",   800))
        self.tile_pad      = int(cfg.get("tile_pad",    10))
        self.use_half      = bool(cfg.get("use_half_precision", True))
        self.io_prefetch   = int(cfg.get("io_prefetch", 12))
        self.write_workers = int(cfg.get("write_workers", 4))
        model_path_str = cfg.get("upscale_model_path", "./weights/RealESRGAN_x4plus.pth")
        if model_path_str.startswith("./"):
            self.model_path = str(Path(__file__).parent / model_path_str[2:])
        else:
            self.model_path = model_path_str
        self.target_w     = int(cfg.get("target_width",  self.TARGET_W))
        self.target_h     = int(cfg.get("target_height", self.TARGET_H))
        self.telemetry_callback = telemetry_callback

        self._device:       str            = self._select_device()
        self._upscaler:     Optional[object] = None   # lazily initialised
        self._optimal_tile: Optional[int]    = None   # set during calibration

        # Thread-safe progress tracking for async telemetry
        self._progress_processed: int = 0
        self._progress_failed:    int = 0
        self._progress_lock = threading.Lock()

    # ── Device Selection ───────────────────────────────────────────────────────

    @staticmethod
    def _select_device() -> str:
        if not TORCH_AVAILABLE:
            return "cpu"
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA device: %s", name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device selected.")
            return "mps"
        logger.info("Using CPU for inference.")
        return "cpu"

    # ── Upscaler Init (Lazy) ───────────────────────────────────────────────────

    def _init_upscaler(self) -> None:
        if self._upscaler is not None:
            return

        if REALESRGAN_AVAILABLE and TORCH_AVAILABLE and Path(self.model_path).exists():
            logger.info(
                "Initialising Real-ESRGAN on %s (tile=%d, half=%s)",
                self._device, self.tile_size, self.use_half,
            )
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=4,
            )
            self._upscaler = RealESRGANer(
                scale=4,
                model_path=self.model_path,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=0,
                half=(self.use_half and self._device == "cuda"),
                device=self._device,
            )

            # ── Optimization: cuDNN benchmark mode ────────────────────────
            # Enables cuDNN auto-tuner to find the fastest convolution
            # algorithms for the current GPU and input dimensions.
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark mode enabled.")

            # ── Optimization: torch.compile ───────────────────────────────
            # Fuses GPU kernels and optimises memory layout.
            # Typically yields 15-30% speedup on RTX 40-series GPUs.
            if hasattr(torch, "compile") and self._device == "cuda":
                try:
                    self._upscaler.model = torch.compile(
                        self._upscaler.model, mode="default"
                    )
                    logger.info("torch.compile applied to Real-ESRGAN model.")
                except Exception as exc:
                    logger.warning("torch.compile unavailable (non-fatal): %s", exc)

        else:
            reason = (
                "Real-ESRGAN weights not found"
                if REALESRGAN_AVAILABLE and not Path(self.model_path).exists()
                else "Real-ESRGAN not installed"
            )
            logger.info("%s – using PyTorch bicubic / OpenCV Lanczos.", reason)
            self._upscaler = None

    # ── Dynamic Tile Calibration ──────────────────────────────────────────────

    def _calibrate_tile_size(self, sample_bgr: np.ndarray) -> None:
        """
        Auto-detect optimal tile size by testing the GPU with a real frame.

        Strategy: try tile=0 (process entire frame at once) first. If the
        GPU has enough VRAM, this eliminates all tiling overhead and is the
        fastest possible mode. On OOM, fall back to the configured tile size.
        """
        if self._upscaler is None:
            self._optimal_tile = self.tile_size
            return

        saved_tile = self._upscaler.tile_size

        # Attempt: No tiling (maximum speed)
        try:
            self._upscaler.tile_size = 0
            logger.info("Calibrating: testing tile=0 (no tiling)...")
            self._upscaler.enhance(sample_bgr, outscale=4)
            self._optimal_tile = 0
            logger.info(
                "Calibration result: tile=0 (NO TILING) – maximum GPU throughput!"
            )
            self._free_vram()
            return
        except (CUDA_OOM_EXCEPTION, RuntimeError) as exc:
            logger.info("Calibration: tile=0 caused OOM (%s)", type(exc).__name__)
            self._free_vram()

        # Fall back to configured tile size
        self._upscaler.tile_size = saved_tile
        self._optimal_tile = saved_tile
        logger.info("Calibration result: tile=%d", saved_tile)

    # ── Public Entry Point ─────────────────────────────────────────────────────

    async def process(
        self,
        frames_dir:        str,
        output_frames_dir: str,
        fps:               float,
        total_frames_hint: int,
    ) -> dict:
        """
        Upscale all frames in `frames_dir` and write results to `output_frames_dir`.

        Parameters
        ----------
        frames_dir        : Directory containing raw extracted PNGs.
        output_frames_dir : Destination directory for 4K PNGs.
        fps               : Video frame rate (informational / telemetry).
        total_frames_hint : Expected total frame count (from FFprobe).

        Returns
        -------
        dict with processing statistics.
        """
        await self._telemetry({"stage": "video_init", "video_progress": 0})

        os.makedirs(output_frames_dir, exist_ok=True)

        frame_paths = sorted(
            p for p in Path(frames_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not frame_paths:
            raise FileNotFoundError(f"No frame images found in: {frames_dir}")

        total = len(frame_paths)
        logger.info(
            "Video upscaling: %d frames, fps=%.2f, device=%s",
            total, fps, self._device,
        )

        # ── Initialise & Calibrate ────────────────────────────────────────
        self._init_upscaler()

        if self._upscaler is not None and frame_paths:
            sample = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
            if sample is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._calibrate_tile_size, sample
                )

        tile_desc = (
            "DISABLED (full-frame)"
            if self._optimal_tile == 0
            else f"tile={self._optimal_tile}"
        )

        await self._telemetry({
            "stage":        "video_upscaling",
            "video_progress": 5,
            "total_frames": total,
            "device":       self._device,
            "upscaler":     "RealESRGAN" if self._upscaler else "bicubic",
            "tiling":       tile_desc,
        })

        # ── Reset progress counters ───────────────────────────────────────
        with self._progress_lock:
            self._progress_processed = 0
            self._progress_failed    = 0

        # ── Launch producer-consumer pipeline in background executor ──────
        loop = asyncio.get_event_loop()
        pipeline_future = loop.run_in_executor(
            None, self._run_pipeline, frame_paths, output_frames_dir
        )

        # ── Async telemetry polling while pipeline runs ───────────────────
        while True:
            done_set, _ = await asyncio.wait({pipeline_future}, timeout=2.0)

            with self._progress_lock:
                p = self._progress_processed
                f = self._progress_failed

            progress = 5 + int((p / max(total, 1)) * 90)
            await self._telemetry({
                "stage":         "video_upscaling",
                "video_progress": progress,
                "frame":         p,
                "total_frames":  total,
                "gpu_util":      self._gpu_memory_percent(),
                "gpu_temp":      self._gpu_temp(),
                "fps":           round(fps, 2),
            })

            if done_set:
                break

        # ── Collect result (raises on pipeline error) ─────────────────────
        processed, failed = pipeline_future.result()

        await self._telemetry({
            "stage":         "video_complete",
            "video_progress": 100,
            "frame":         processed,
            "total_frames":  total,
        })

        return {
            "frames_processed": processed,
            "frames_failed":    failed,
            "total_frames":     total,
            "device":           self._device,
        }

    # ── Producer-Consumer Pipeline ─────────────────────────────────────────────

    def _run_pipeline(
        self,
        frame_paths: List[Path],
        out_dir:     str,
    ) -> Tuple[int, int]:
        """
        High-throughput frame processing pipeline.

        Architecture
        ------------
        ┌──────────┐     ┌──────────────┐     ┌──────────────┐
        │  Reader  │ ──► │  GPU Worker   │ ──► │  Writer Pool │
        │ (thread) │     │ (main thread) │     │ (4 threads)  │
        └──────────┘     └──────────────┘     └──────────────┘
             ↓                                       ↓
          read_q                               ThreadPoolExecutor
        (prefetch 12)                          (parallel PNG write)

        The reader constantly loads frames into RAM so the GPU never
        waits for disk I/O.  The writer pool compresses and flushes
        upscaled PNGs across multiple CPU cores in parallel.

        Returns
        -------
        (processed, failed) tuple.
        """
        read_q:      queue.Queue = queue.Queue(maxsize=self.io_prefetch)
        read_done:   threading.Event = threading.Event()
        read_errors: List[Path] = []

        write_pool = ThreadPoolExecutor(
            max_workers=self.write_workers,
            thread_name_prefix="frame-writer",
        )
        write_futures: list = []

        # ── Reader thread ─────────────────────────────────────────────────
        def _reader() -> None:
            try:
                for fp in frame_paths:
                    bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                    if bgr is None:
                        logger.error("cv2.imread returned None: %s", fp)
                        read_errors.append(fp)
                        continue
                    read_q.put((fp, bgr))
            except Exception as exc:
                logger.error("Reader thread error: %s", exc)
            finally:
                read_done.set()

        reader_t = threading.Thread(
            target=_reader, daemon=True, name="frame-reader"
        )
        reader_t.start()

        # ── GPU processing loop ───────────────────────────────────────────
        processed = 0
        failed    = 0

        try:
            while True:
                # Pull next frame from pre-loaded queue
                try:
                    fp, bgr = read_q.get(timeout=3.0)
                except queue.Empty:
                    if read_done.is_set() and read_q.empty():
                        break
                    continue

                # Upscale with AI model
                try:
                    upscaled = self._upscale_frame(bgr)

                    # Hard-cap output to target 4K dimensions
                    h, w = upscaled.shape[:2]
                    if w > self.target_w or h > self.target_h:
                        upscaled = cv2.resize(
                            upscaled,
                            (self.target_w, self.target_h),
                            interpolation=cv2.INTER_LANCZOS4,
                        )

                    # Submit write to thread pool (non-blocking)
                    out_path = str(Path(out_dir) / fp.name)
                    fut = write_pool.submit(
                        cv2.imwrite, out_path, upscaled,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1],
                    )
                    write_futures.append(fut)
                    processed += 1

                except CUDA_OOM_EXCEPTION:
                    logger.warning(
                        "VRAM OOM on frame %s – bicubic fallback", fp.name
                    )
                    self._free_vram()
                    try:
                        upscaled = self._fallback_upscale(bgr)
                        out_path = str(Path(out_dir) / fp.name)
                        fut = write_pool.submit(
                            cv2.imwrite, out_path, upscaled,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1],
                        )
                        write_futures.append(fut)
                        processed += 1
                    except Exception:
                        self._copy_frame_as_is(fp, out_dir)
                        failed += 1

                except Exception as exc:
                    logger.error("Frame %s error: %s", fp.name, exc)
                    try:
                        upscaled = self._fallback_upscale(bgr)
                        out_path = str(Path(out_dir) / fp.name)
                        fut = write_pool.submit(
                            cv2.imwrite, out_path, upscaled,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1],
                        )
                        write_futures.append(fut)
                        processed += 1
                    except Exception:
                        self._copy_frame_as_is(fp, out_dir)
                        failed += 1

                # Update thread-safe progress counters
                with self._progress_lock:
                    self._progress_processed = processed
                    self._progress_failed    = failed

                # Prune completed write futures to prevent unbounded list growth
                if len(write_futures) > 50:
                    write_futures = [f for f in write_futures if not f.done()]

            # Count frames that failed to load from disk
            failed += len(read_errors)
            for fp in read_errors:
                self._copy_frame_as_is(fp, out_dir)

        finally:
            # Drain all pending writes
            write_pool.shutdown(wait=True)
            reader_t.join(timeout=10)

            # Final progress update
            with self._progress_lock:
                self._progress_processed = processed
                self._progress_failed    = failed

        return processed, failed

    # ── Frame Upscaling ────────────────────────────────────────────────────────

    def _upscale_frame(self, bgr: np.ndarray) -> np.ndarray:
        """Run the primary upscaler (Real-ESRGAN or fallback) on a single frame."""
        if self._upscaler is not None:
            result, _ = self._upscaler.enhance(bgr, outscale=4)
            return result
        elif TORCH_AVAILABLE:
            return self._torch_bicubic(bgr)
        else:
            return self._cv2_lanczos(bgr)

    def _fallback_upscale(self, bgr: np.ndarray) -> np.ndarray:
        """Bicubic / Lanczos fallback when ESRGAN fails on a frame."""
        if TORCH_AVAILABLE:
            return self._torch_bicubic(bgr)
        return self._cv2_lanczos(bgr)

    # ── Per-Frame Upscaling Implementations ───────────────────────────────────

    def _torch_bicubic(self, bgr: np.ndarray) -> np.ndarray:
        """
        PyTorch bicubic upscaling to 4×, capped at target_w × target_h.
        Runs on the configured device (CUDA / MPS / CPU).
        """
        h, w = bgr.shape[:2]
        target_h = min(h * 4, self.target_h)
        target_w = min(w * 4, self.target_w)

        # BGR → RGB → float32 → (1, 3, H, W) tensor
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t     = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        if self._device != "cpu":
            t = t.to(self._device)

        with torch.no_grad():
            up = F.interpolate(
                t,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)

        out_np = (up.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0) \
                     .round().astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def _cv2_lanczos(self, bgr: np.ndarray) -> np.ndarray:
        """OpenCV Lanczos-4 fallback (CPU only)."""
        h, w    = bgr.shape[:2]
        target_w = min(w * 4, self.target_w)
        target_h = min(h * 4, self.target_h)
        return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def _copy_frame_as_is(self, frame_path: Path, out_dir: str) -> None:
        """Last-resort fallback: copy source frame resized to target dimensions so remux can proceed."""
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            import shutil
            shutil.copy2(str(frame_path), str(Path(out_dir) / frame_path.name))
            return
        h, w = bgr.shape[:2]
        t_w = min(w * 4, self.target_w)
        t_h = min(h * 4, self.target_h)
        resized = cv2.resize(bgr, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(Path(out_dir) / frame_path.name), resized, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # ── VRAM / Memory Management ───────────────────────────────────────────────

    def _free_vram(self) -> None:
        """Release GPU memory (used for OOM recovery and calibration cleanup)."""
        if TORCH_AVAILABLE and self._device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _gpu_memory_percent(self) -> float:
        """Return GPU memory utilisation as a percentage (0–100)."""
        if not TORCH_AVAILABLE or self._device != "cuda":
            return 0.0
        try:
            alloc = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return round(alloc / max(total, 1) * 100.0, 1)
        except Exception:
            return 0.0

    @staticmethod
    def _gpu_temp() -> float:
        """Return GPU temperature in °C via nvidia-smi (best-effort)."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return 0.0

    # ── Telemetry ──────────────────────────────────────────────────────────────

    async def _telemetry(self, data: dict) -> None:
        if self.telemetry_callback is not None:
            try:
                await self.telemetry_callback(data)
            except Exception as exc:
                logger.debug("Telemetry callback error (non-fatal): %s", exc)
