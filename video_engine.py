"""
video_engine.py – AV-SynthRestore 3D
4K video upscaling using Real-ESRGAN via TensorRT/ONNX (or PyTorch bicubic
fallback) with optimized GPU pipeline: producer-consumer I/O and direct
NVENC hardware video encoding.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import gc
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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
    import torchvision
    import torchvision.transforms.functional
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional
except ImportError:
    pass

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    logger.info("Real-ESRGAN detected.")
except ImportError as exc:
    REALESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN not installed; bicubic fallback active. Error: %s", exc)


# ── VideoEngine ────────────────────────────────────────────────────────────────

class VideoEngine:
    """
    Orchestrates per-frame AI upscaling to 4K.

    Processing Pipeline (TensorRT + NVENC)
    ---------------------------------------
    1. Load Real-ESRGAN model via ONNX Runtime with TensorRT execution
       provider for hardware-optimised inference.
    2. Producer-Consumer I/O pipeline:
       - Reader thread:  pre-loads frames from disk into a RAM queue.
       - GPU thread:     pulls from read queue, runs TensorRT inference.
       - FFmpeg pipe:    streams raw frames into NVENC hardware encoder.
    3. Per-frame OOM fallback to PyTorch bicubic / OpenCV Lanczos.
    4. Async telemetry updates every 2 seconds while pipeline runs.
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
        self.tile_size     = int(cfg.get("tile_size",   500))
        self.tile_pad      = int(cfg.get("tile_pad",    10))
        self.use_half      = bool(cfg.get("use_half_precision", True))
        self.io_prefetch   = int(cfg.get("io_prefetch", 3))
        self.write_workers = int(cfg.get("write_workers", 2))
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

        # Optimization: Enable cuDNN benchmark for faster convolutions
        if TORCH_AVAILABLE and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled.")

        if REALESRGAN_AVAILABLE and TORCH_AVAILABLE and Path(self.model_path).exists():
            logger.info(
                "Loading PyTorch Real-ESRGAN on %s (tile=%d)",
                self._device, self.tile_size,
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
                device=torch.device('cuda' if self._device == 'cuda' else 'cpu')
            )
            logger.info("Real-ESRGAN PyTorch loaded.")
        else:
            reason = (
                "Real-ESRGAN weights not found"
                if REALESRGAN_AVAILABLE
                else "Real-ESRGAN not installed"
            )
            logger.info("%s – using PyTorch bicubic / OpenCV Lanczos.", reason)
            self._upscaler = None

    # ── Dynamic Tile Calibration ──────────────────────────────────────────────

    def _calibrate_tile_size(self, sample_bgr: np.ndarray) -> None:
        """
        Auto-detect optimal tile size. We bypass tile=0 testing on Windows 
        because it tends to hang PyTorch (swaps to system RAM instead of OOM).
        We just use the configured tile_size.
        """
        if self._upscaler is None:
            self._optimal_tile = self.tile_size
            return

        self._optimal_tile = self.tile_size
        self._upscaler.tile_size = self._optimal_tile
        logger.info("Using configured tile size: %d", self._optimal_tile)

    # ── Public Entry Point ─────────────────────────────────────────────────────

    async def process(
        self,
        frames_dir:        str,
        output_video_path: str,
        fps:               float,
        total_frames_hint: int,
    ) -> dict:
        """
        Upscale all frames in `frames_dir` and stream results directly to 
        `output_video_path` using an NVENC FFmpeg pipe.

        Parameters
        ----------
        frames_dir        : Directory containing raw extracted PNGs.
        output_video_path : Destination MP4 file path for the NVENC stream.
        fps               : Video frame rate.
        total_frames_hint : Expected total frame count (from FFprobe).

        Returns
        -------
        dict with processing statistics.
        """
        await self._telemetry({"stage": "video_init", "video_progress": 0})

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
            None, self._run_pipeline, frame_paths, output_video_path, fps
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
                "gpu_temp":      await self._gpu_temp(),
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
        output_video_path: str,
        fps: float,
    ) -> Tuple[int, int]:
        """
        High-throughput NVENC direct-to-video pipeline.

        Architecture
        ------------
        ┌──────────┐     ┌───────────────┐     ┌───────────────┐
        │  Reader  │ ──► │  GPU Worker   │ ──► │  FFmpeg Pipe  │
        │ (pool)   │     │ (main thread) │     │ (writer thr)  │
        └──────────┘     └───────────────┘     └───────────────┘

        Returns
        -------
        (processed, failed) tuple.
        """
        read_q:      queue.Queue = queue.Queue(maxsize=self.io_prefetch * 2)
        write_q:     queue.Queue = queue.Queue(maxsize=self.io_prefetch)
        read_errors: List[Path] = []

        # ── FFmpeg NVENC Pipe ─────────────────────────────────────────────
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.target_w}x{self.target_h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-", # read from stdin
            "-c:v", "h264_nvenc",
            "-preset", "p6",
            "-cq", "19",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=10**8
        )

        # ── Writer Thread ─────────────────────────────────────────────────
        def _writer() -> None:
            try:
                while True:
                    frame_bytes = write_q.get()
                    if frame_bytes is None: # Sentinel
                        break
                    ffmpeg_proc.stdin.write(frame_bytes)
            except Exception as exc:
                logger.error("Writer thread error: %s", exc)
                
        writer_t = threading.Thread(target=_writer, daemon=True, name="frame-writer")
        writer_t.start()

        # ── Reader Thread (Parallel Fetching) ──────────────────────────────
        def _reader() -> None:
            try:
                # Use threads to parallelise I/O while preserving order
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as pool:
                    # Submit in small chunks to prevent OOM
                    chunk_size = max(1, self.io_prefetch)
                    for i in range(0, len(frame_paths), chunk_size):
                        chunk = frame_paths[i:i+chunk_size]
                        futures = [pool.submit(cv2.imread, str(fp), cv2.IMREAD_COLOR) for fp in chunk]
                        for fp, fut in zip(chunk, futures):
                            bgr = fut.result()
                            if bgr is None:
                                logger.error("cv2.imread returned None: %s", fp)
                                read_errors.append(fp)
                                continue
                            read_q.put((fp, bgr))
            except Exception as exc:
                logger.error("Reader thread error: %s", exc)
            finally:
                read_q.put(None) # Sentinel

        reader_t = threading.Thread(target=_reader, daemon=True, name="frame-reader")
        reader_t.start()

        # ── GPU processing loop ───────────────────────────────────────────
        processed = 0
        failed    = 0

        try:
            while True:
                item = read_q.get()
                if item is None:
                    break
                fp, bgr = item

                # Upscale with AI model
                try:
                    upscaled = self._upscale_frame(bgr)

                    # Hard-cap output to target 4K dimensions
                    h, w = upscaled.shape[:2]
                    if w > self.target_w or h > self.target_h:
                        upscaled = cv2.resize(
                            upscaled,
                            (self.target_w, self.target_h),
                            interpolation=cv2.INTER_AREA,
                        )

                    # Stream directly into writer queue (avoids GPU stall)
                    write_q.put(upscaled.tobytes())
                    processed += 1

                except CUDA_OOM_EXCEPTION:
                    logger.warning("VRAM OOM on frame %s – bicubic fallback", fp.name)
                    self._free_vram()
                    try:
                        upscaled = self._fallback_upscale(bgr)
                        write_q.put(upscaled.tobytes())
                        processed += 1
                    except Exception:
                        failed += 1

                except Exception as exc:
                    logger.error("Frame %s error: %s", fp.name, exc)
                    try:
                        upscaled = self._fallback_upscale(bgr)
                        write_q.put(upscaled.tobytes())
                        processed += 1
                    except Exception:
                        failed += 1

                # Update thread-safe progress counters
                with self._progress_lock:
                    self._progress_processed = processed
                    self._progress_failed    = failed

            # Count frames that failed to load from disk
            failed += len(read_errors)

        finally:
            # Signal writer to exit and wait
            write_q.put(None)
            writer_t.join(timeout=10)
            
            # Drain and close the video pipe
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
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
            with suppress_stdout():
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
        Optimized to avoid unnecessary CPU memory allocations and color conversions.
        """
        h, w = bgr.shape[:2]
        target_h = min(h * 4, self.target_h)
        target_w = min(w * 4, self.target_w)

        # Zero-copy view of numpy array -> contiguous tensor -> format (1, 3, H, W)
        t = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0)
        
        # Async transfer to device (if GPU), convert to float inline (PyTorch bicubic requires float)
        if self._device != "cpu":
            t = t.to(self._device, non_blocking=True).float()
        else:
            t = t.float()

        with torch.no_grad():
            # Upscale directly on raw 0-255 values
            up = F.interpolate(
                t,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            ).clamp(0, 255)

        # Byte cast directly on GPU, then synchronous pull to CPU, format (H, W, 3)
        return up.squeeze(0).byte().cpu().numpy().transpose(1, 2, 0)

    def _cv2_lanczos(self, bgr: np.ndarray) -> np.ndarray:
        """OpenCV Cubic fallback (CPU only) - optimized for speed over Lanczos."""
        h, w    = bgr.shape[:2]
        target_w = min(w * 4, self.target_w)
        target_h = min(h * 4, self.target_h)
        return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

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
    async def _gpu_temp() -> float:
        """Return GPU temperature in °C via nvidia-smi (best-effort, non-blocking)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            if proc.returncode == 0 and stdout:
                return float(stdout.decode().strip().split("\n")[0])
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
