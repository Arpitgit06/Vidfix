"""
video_engine.py – AV-SynthRestore 3D
4K video upscaling using Real-ESRGAN (or PyTorch bicubic fallback) with
VRAM-safe batch tensor processing and per-frame CUDA memory cleanup.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
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

    Processing stages
    -----------------
    1. Enumerate extracted frame PNGs from `frames_dir`.
    2. Initialise Real-ESRGAN model (lazy, first use only).
    3. Process frames in configurable batches.
       – Each batch is run off the event loop so WebSocket keepalives continue.
       – After every batch: `torch.cuda.empty_cache()` + `gc.collect()`.
    4. Per-frame fallback to PyTorch bicubic / OpenCV Lanczos on ESRGAN error.
    5. Write upscaled frames to `output_frames_dir` as lossless PNG.
    """

    TARGET_W: int = 3840
    TARGET_H: int = 2160

    def __init__(
        self,
        config:   Optional[dict]     = None,
        telemetry_callback: Optional[Callable] = None,
    ) -> None:
        cfg = config or {}
        self.batch_size   = int(cfg.get("batch_size",  4))
        self.tile_size    = int(cfg.get("tile_size",   256))
        self.tile_pad     = int(cfg.get("tile_pad",    10))
        self.use_half     = bool(cfg.get("use_half_precision", True))
        model_path_str = cfg.get("upscale_model_path", "./weights/RealESRGAN_x4plus.pth")
        if model_path_str.startswith("./"):
            self.model_path = str(Path(__file__).parent / model_path_str[2:])
        else:
            self.model_path = model_path_str
        self.target_w     = int(cfg.get("target_width",  self.TARGET_W))
        self.target_h     = int(cfg.get("target_height", self.TARGET_H))
        self.telemetry_callback = telemetry_callback

        self._device:   str          = self._select_device()
        self._upscaler: Optional[object] = None   # lazily initialised

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
        else:
            reason = (
                "Real-ESRGAN weights not found"
                if REALESRGAN_AVAILABLE and not Path(self.model_path).exists()
                else "Real-ESRGAN not installed"
            )
            logger.info("%s – using PyTorch bicubic / OpenCV Lanczos.", reason)
            self._upscaler = None

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

        self._init_upscaler()

        await self._telemetry({
            "stage":        "video_upscaling",
            "video_progress": 5,
            "total_frames": total,
            "device":       self._device,
            "upscaler":     "RealESRGAN" if self._upscaler else "bicubic",
        })

        # ── Batch Loop ────────────────────────────────────────────────────
        batches    = self._make_batches(frame_paths, self.batch_size)
        processed  = 0
        failed     = 0

        for batch_idx, batch in enumerate(batches):
            try:
                await self._process_batch(batch, output_frames_dir)
                processed += len(batch)
            except CUDA_OOM_EXCEPTION as oom:
                logger.warning(
                    "VRAM OOM on batch %d – falling back to frame-by-frame bicubic: %s",
                    batch_idx, oom,
                )
                self._free_vram()
                for fp in batch:
                    try:
                        self._bicubic_single(fp, output_frames_dir)
                        processed += 1
                    except Exception as fe:
                        logger.error("Frame %s failed even on fallback: %s", fp.name, fe)
                        # Copy source frame as-is so the remux doesn't break
                        self._copy_frame_as_is(fp, output_frames_dir)
                        failed += 1
            except Exception as exc:
                logger.error("Batch %d error: %s", batch_idx, exc)
                for fp in batch:
                    try:
                        self._bicubic_single(fp, output_frames_dir)
                        processed += 1
                    except Exception:
                        self._copy_frame_as_is(fp, output_frames_dir)
                        failed += 1
            finally:
                self._free_vram()

            progress   = 5 + int((processed / total) * 90)
            gpu_util   = self._gpu_memory_percent()
            gpu_temp   = self._gpu_temp()

            await self._telemetry({
                "stage":         "video_upscaling",
                "video_progress": progress,
                "frame":         processed,
                "total_frames":  total,
                "gpu_util":      gpu_util,
                "gpu_temp":      gpu_temp,
                "batch":         batch_idx + 1,
                "total_batches": len(batches),
                "fps":           round(fps, 2),
            })

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

    # ── Batch Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_batches(
        paths: List[Path], batch_size: int
    ) -> List[List[Path]]:
        return [paths[i: i + batch_size] for i in range(0, len(paths), batch_size)]

    async def _process_batch(
        self, batch: List[Path], out_dir: str
    ) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_process_batch, batch, out_dir)

    def _sync_process_batch(self, batch: List[Path], out_dir: str) -> None:
        for frame_path in batch:
            out_path = Path(out_dir) / frame_path.name
            bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"cv2.imread returned None for: {frame_path}")

            if self._upscaler is not None:
                upscaled, _ = self._upscaler.enhance(bgr, outscale=4)
            elif TORCH_AVAILABLE:
                upscaled = self._torch_bicubic(bgr)
            else:
                upscaled = self._cv2_lanczos(bgr)

            # Hard-cap output to target 4K dimensions
            h, w = upscaled.shape[:2]
            if w > self.target_w or h > self.target_h:
                upscaled = cv2.resize(
                    upscaled,
                    (self.target_w, self.target_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Lossless PNG with fast compression (level 1)
            cv2.imwrite(str(out_path), upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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

    def _bicubic_single(self, frame_path: Path, out_dir: str) -> None:
        """Emergency per-frame bicubic upscale (fallback from batch OOM)."""
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(str(frame_path))
        out_path = Path(out_dir) / frame_path.name
        if TORCH_AVAILABLE:
            upscaled = self._torch_bicubic(bgr)
        else:
            upscaled = self._cv2_lanczos(bgr)
        cv2.imwrite(str(out_path), upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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
        """Release GPU memory after each batch."""
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
