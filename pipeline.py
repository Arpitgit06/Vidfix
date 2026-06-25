"""
pipeline.py – AV-SynthRestore 3D
Top-level orchestration of the full restoration pipeline:

    Input File
        │
        ▼
    FFmpeg Demux  (audio WAV + frame PNGs extracted in parallel)
        │
    ┌───┴───────────────┐
    │                   │
    ▼                   ▼
AudioEngine         VideoEngine
(gap-fill +         (4K upscale)
 denoise)
    │                   │
    └───────┬───────────┘
            │
            ▼
       FFmpeg Remux  →  restored_<name>.mp4
            │
            ▼
       Cleanup temp files
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ffmpeg

from audio_engine import AudioEngine
from video_engine import VideoEngine

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


# ── ProcessingPipeline ─────────────────────────────────────────────────────────

class ProcessingPipeline:
    """
    Runs the end-to-end AV restoration pipeline for a single job.

    Usage
    -----
    pipeline = ProcessingPipeline(job_id, connection_manager)
    await pipeline.run()
    """

    def __init__(
        self,
        job_id:             str,
        connection_manager: Any,
        config:             Optional[Dict] = None,
    ) -> None:
        self.job_id  = job_id
        self.manager = connection_manager
        self.config  = config or self._load_config()

        self.job_dir              = Path(f"./jobs/{job_id}")
        self.temp_dir             = self.job_dir / "temp"
        self.frames_dir           = self.temp_dir / "frames"
        self.upscaled_frames_dir  = self.temp_dir / "upscaled_frames"
        self.output_dir           = Path(
            self.config["system"].get("output_directory", "./restored_output")
        )
        self._start_time: Optional[float] = None

    # ── Config ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config() -> Dict:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "system": {"acceleration": "auto", "threads": 8,
                       "output_directory": "./restored_output"},
            "audio":  {"noise_reduction_aggressive_level": 0.5},
            "video":  {"target_resolution": "4K", "batch_size": 4},
        }

    # ── Public Entry Point ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Execute the pipeline; sends telemetry payloads throughout."""
        self._start_time = time.monotonic()

        try:
            self._prepare_directories()
            await self._tele({"stage": "pipeline_start", "overall_progress": 0})

            # ── Probe + Validate ──────────────────────────────────────────
            input_path = self._find_input_file()
            await self._tele({
                "stage": "input_loaded", "overall_progress": 3,
                "input_file": input_path.name,
            })

            probe      = self._probe(input_path)
            video_info = self._stream_info(probe, "video")
            audio_info = self._stream_info(probe, "audio")
            has_video  = video_info is not None
            has_audio  = audio_info is not None

            if not has_video and not has_audio:
                raise ValueError("Input contains no usable video or audio streams.")

            fps          = self._parse_fps(video_info.get("r_frame_rate", "24/1") if has_video else "0/1")
            total_frames = int(video_info.get("nb_frames", 0)) if has_video else 0
            duration     = float(probe.get("format", {}).get("duration", 0.0))

            logger.info(
                "Probed: has_video=%s has_audio=%s fps=%.2f frames=%d dur=%.2fs",
                has_video, has_audio, fps, total_frames, duration,
            )
            await self._tele({
                "stage": "probed", "overall_progress": 6,
                "fps": round(fps, 2), "total_frames": total_frames,
                "duration": round(duration, 2),
                "has_audio": has_audio, "has_video": has_video,
            })

            # ── Stage 1: Demux ────────────────────────────────────────────
            await self._tele({"stage": "demuxing", "overall_progress": 10})

            raw_audio_path: Optional[Path] = None
            if has_audio:
                raw_audio_path = self.temp_dir / "raw_audio.wav"
                await self._run_in_executor(
                    self._ffmpeg_extract_audio, input_path, raw_audio_path
                )

            if has_video:
                await self._tele({"stage": "frame_extraction", "overall_progress": 14})
                await self._run_in_executor(
                    self._ffmpeg_extract_frames, input_path
                )
                extracted = len(list(self.frames_dir.glob("*.png")))
                total_frames = extracted if extracted > 0 else total_frames

            await self._tele({
                "stage": "demuxed", "overall_progress": 22,
                "total_frames": total_frames,
            })

            # ── Stage 2: Parallel AI Processing ──────────────────────────
            await self._tele({
                "stage": "parallel_processing_start", "overall_progress": 25,
            })

            audio_result: Optional[Dict] = None
            video_result: Optional[Dict] = None

            tasks = []

            if has_audio and raw_audio_path is not None:
                audio_engine = AudioEngine(
                    config=self.config.get("audio"),
                    telemetry_callback=self._make_branch_callback("audio"),
                )
                repaired_audio = self.temp_dir / "repaired_audio.wav"
                tasks.append(("audio", audio_engine.process(
                    str(raw_audio_path), str(repaired_audio)
                )))

            if has_video:
                video_engine = VideoEngine(
                    config=self.config.get("video"),
                    telemetry_callback=self._make_branch_callback("video"),
                )
                tasks.append(("video", video_engine.process(
                    str(self.frames_dir),
                    str(self.upscaled_frames_dir),
                    fps,
                    total_frames,
                )))

            # Run both branches truly in parallel
            coros      = [t[1] for t in tasks]
            labels     = [t[0] for t in tasks]
            results    = await asyncio.gather(*coros, return_exceptions=True)

            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    logger.error("%s branch failed: %s", label, result)
                    raise result
                if label == "audio":
                    audio_result = result
                else:
                    video_result = result

            gc.collect()
            await self._tele({
                "stage": "processing_complete", "overall_progress": 85,
                "audio_result": audio_result,
                "video_result": video_result,
            })

            # ── Stage 3: Remux ────────────────────────────────────────────
            await self._tele({"stage": "remuxing", "overall_progress": 88})

            output_stem = f"restored_{input_path.stem}"
            output_path = self.output_dir / f"{output_stem}.mp4"

            await self._run_in_executor(
                self._ffmpeg_remux,
                has_video,
                has_audio,
                self.temp_dir / "repaired_audio.wav" if has_audio else None,
                fps,
                str(output_path),
            )

            elapsed       = time.monotonic() - self._start_time
            file_size_mb  = output_path.stat().st_size / (1024 ** 2) \
                            if output_path.exists() else 0.0

            await self._tele({
                "stage":                 "pipeline_complete",
                "overall_progress":       100,
                "output_file":           str(output_path),
                "elapsed_seconds":       round(elapsed, 1),
                "output_size_mb":        round(file_size_mb, 2),
                "audio_gaps_repaired":   audio_result.get("gaps_repaired", 0)
                                         if audio_result else 0,
                "video_frames_processed": video_result.get("frames_processed", 0)
                                          if video_result else 0,
            })

        except Exception as exc:
            elapsed = time.monotonic() - (self._start_time or time.monotonic())
            logger.exception("Pipeline failed for job %s: %s", self.job_id, exc)
            await self._tele({
                "stage":            "error",
                "error":            str(exc),
                "overall_progress": -1,
                "elapsed_seconds":  round(elapsed, 1),
            })
        finally:
            await self._cleanup()

    # ── Directory Setup ────────────────────────────────────────────────────────

    def _prepare_directories(self) -> None:
        for d in (
            self.temp_dir,
            self.frames_dir,
            self.upscaled_frames_dir,
            self.output_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ── File Discovery ─────────────────────────────────────────────────────────

    def _find_input_file(self) -> Path:
        for f in self.job_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS:
                return f
        raise FileNotFoundError(
            f"No supported video file found in {self.job_dir}. "
            f"Accepted: {', '.join(sorted(_SUPPORTED_EXTS))}"
        )

    # ── FFprobe ────────────────────────────────────────────────────────────────

    @staticmethod
    def _probe(input_path: Path) -> Dict:
        try:
            return ffmpeg.probe(str(input_path))
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"FFprobe failed on {input_path.name}: "
                f"{exc.stderr.decode(errors='replace')}"
            ) from exc

    @staticmethod
    def _stream_info(probe: Dict, codec_type: str) -> Optional[Dict]:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == codec_type:
                return stream
        return None

    @staticmethod
    def _parse_fps(fps_str: str) -> float:
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den) if float(den) else 24.0
            return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 24.0

    # ── FFmpeg Operations ──────────────────────────────────────────────────────

    def _ffmpeg_extract_audio(self, input_path: Path, out_path: Path) -> None:
        """Extract audio to 24-bit 48 kHz PCM WAV."""
        try:
            (
                ffmpeg
                .input(str(input_path))
                .output(
                    str(out_path),
                    acodec="pcm_s24le",
                    ar=48000,
                    vn=None,        # no video stream
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"Audio extraction failed: {exc.stderr.decode(errors='replace')}"
            ) from exc

    def _ffmpeg_extract_frames(self, input_path: Path) -> None:
        """Demux all video frames to sequentially numbered PNGs."""
        pattern = str(self.frames_dir / "frame_%06d.png")
        try:
            (
                ffmpeg
                .input(str(input_path))
                .output(pattern, vsync="0", an=None)  # no audio stream
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"Frame extraction failed: {exc.stderr.decode(errors='replace')}"
            ) from exc

    def _ffmpeg_remux(
        self,
        has_video:    bool,
        has_audio:    bool,
        audio_path:   Optional[Path],
        fps:          float,
        output_path:  str,
    ) -> None:
        """
        Combine upscaled frames (PNG sequence) + repaired WAV → H.264/AAC MP4.
        Handles video-only, audio-only, and combined cases.
        """
        video_enc_opts = dict(
            vcodec="libx264",
            crf=self.config["video"].get("encoder_crf", 18),
            preset=self.config["video"].get("encoder_preset", "slow"),
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        audio_enc_opts = dict(
            acodec="aac",
            audio_bitrate="320k",
        )

        try:
            if has_video and has_audio and audio_path:
                frame_pattern = str(self.upscaled_frames_dir / "frame_%06d.png")
                vid_in  = ffmpeg.input(
                    frame_pattern, framerate=fps, pattern_type="sequence"
                )
                aud_in  = ffmpeg.input(str(audio_path))
                (
                    ffmpeg
                    .output(vid_in, aud_in, output_path,
                            **video_enc_opts, **audio_enc_opts)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )

            elif has_video:
                frame_pattern = str(self.upscaled_frames_dir / "frame_%06d.png")
                vid_in = ffmpeg.input(
                    frame_pattern, framerate=fps, pattern_type="sequence"
                )
                (
                    ffmpeg
                    .output(vid_in, output_path, **video_enc_opts, an=None)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )

            elif has_audio and audio_path:
                aud_in = ffmpeg.input(str(audio_path))
                (
                    ffmpeg
                    .output(aud_in, output_path, **audio_enc_opts, vn=None)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )

        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"Remux failed: {exc.stderr.decode(errors='replace')}"
            ) from exc

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    async def _run_in_executor(fn, *args) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    def _make_branch_callback(self, branch: str) -> Any:
        """Return an async callable that injects the branch tag into telemetry."""
        async def _cb(data: Dict) -> None:
            await self._tele({**data, "branch": branch})
        return _cb

    async def _tele(self, data: Dict) -> None:
        payload = {
            "job_id":    self.job_id,
            "timestamp": time.time(),
            **data,
        }
        await self.manager.send_telemetry(self.job_id, payload)
        logger.debug("[TELE] %s", payload)

    async def _cleanup(self) -> None:
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temp dir for job %s", self.job_id)
        except Exception as exc:
            logger.warning("Cleanup error for job %s: %s", self.job_id, exc)
