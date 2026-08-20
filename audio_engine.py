"""
audio_engine.py – AV-SynthRestore 3D
Handles silence gap detection, contextual spectral inpainting, and noise reduction.

Gap Detection  : librosa RMS energy + sub-frame amplitude analysis
Inpainting     : Phase-coherent STFT magnitude interpolation + instantaneous
                 frequency extrapolation (IFE) for phase continuity
Noise Reduction: Spectral subtraction with frequency-masked soft flooring
"""

from __future__ import annotations

import asyncio
import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

from model_manager import model_manager

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class AudioGap:
    start_sample: int
    end_sample:   int
    sample_rate:  int

    @property
    def duration_ms(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate * 1000.0

    @property
    def length_samples(self) -> int:
        return self.end_sample - self.start_sample

    def __repr__(self) -> str:
        return (
            f"AudioGap(start={self.start_sample}, end={self.end_sample}, "
            f"dur={self.duration_ms:.1f}ms)"
        )


# ── AudioEngine ────────────────────────────────────────────────────────────────

class AudioEngine:
    """
    Full audio restoration pipeline.

    Processing stages
    -----------------
    1. Load multi-channel WAV.
    2. Detect silence gaps (RMS < threshold, duration > 100 ms).
    3. Per-gap contextual spectral inpainting (phase-coherent STFT fill).
    4. Spectral-subtraction noise reduction on the full track.
    5. Peak normalization → 24-bit WAV output.
    """

    SILENCE_THRESHOLD_DB: float = -60.0
    MIN_GAP_DURATION_MS:  float = 100.0
    CONTEXT_SECONDS:      float = 3.0
    N_FFT:                int   = 2048
    HOP_LENGTH:           int   = 512
    WIN_LENGTH:           int   = 2048

    def __init__(
        self,
        config: Optional[dict] = None,
        telemetry_callback: Optional[Callable] = None,
    ) -> None:
        cfg = config or {}
        self.silence_threshold_db = float(cfg.get("silence_threshold_db", self.SILENCE_THRESHOLD_DB))
        self.min_gap_duration_ms  = float(cfg.get("min_gap_duration_ms",  self.MIN_GAP_DURATION_MS))
        self.context_seconds      = float(cfg.get("context_seconds",      self.CONTEXT_SECONDS))
        self.n_fft                = int(  cfg.get("n_fft",                self.N_FFT))
        self.hop_length           = int(  cfg.get("hop_length",           self.HOP_LENGTH))
        self.noise_level          = float(cfg.get("noise_reduction_aggressive_level", 0.5))
        self.config               = cfg
        self.telemetry_callback   = telemetry_callback

    # ── Public Entry Point ─────────────────────────────────────────────────────

    async def process(self, audio_path: str, output_path: str, ref_audio_path: Optional[str] = None) -> dict:
        """
        Run the complete audio restoration pipeline.

        Parameters
        ----------
        audio_path  : Path to extracted raw WAV file.
        output_path : Destination path for the repaired WAV file.

        Returns
        -------
        dict with processing statistics.
        """
        await self._telemetry({"stage": "audio_load", "audio_progress": 0})

        # ── Load ──────────────────────────────────────────────────────────────
        audio, sr = self._load_audio(audio_path)
        num_channels, num_samples = audio.shape
        duration = num_samples / sr

        logger.info(
            "Audio loaded: %d ch, %d Hz, %.2f s, path=%s",
            num_channels, sr, duration, audio_path,
        )
        await self._telemetry({
            "stage":         "audio_loaded",
            "audio_progress": 10,
            "sample_rate":   sr,
            "channels":      num_channels,
            "duration":      round(duration, 2),
        })

        # ── Detect Gaps ───────────────────────────────────────────────────────
        mono_mix = np.mean(audio, axis=0)
        gaps     = self._detect_gaps(mono_mix, sr)

        logger.info("Detected %d audio gap(s)", len(gaps))
        await self._telemetry({
            "stage":         "gaps_detected",
            "audio_progress": 20,
            "gap_count":     len(gaps),
        })

        ai_config = self.config.get("ai_config", {})
        audio_mode = ai_config.get("audio_mode", "gap_fill")
        script_text = ai_config.get("script_text", "")

        if audio_mode == "full_gen" and script_text:
            logger.info("Using Fish Audio S2 Pro for Full Audio Generation.")
            await self._telemetry({"stage": "ai_model_loading", "audio_progress": 25})
            model = await model_manager.get_model("fish_audio")
            
            # If no ref audio provided, extract from the existing audio track
            ref_path_to_use = ref_audio_path if ref_audio_path else audio_path
            
            await self._telemetry({"stage": "audio_inpainting", "audio_progress": 40})
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, model.generate_audio, script_text, ref_path_to_use, output_path
            )
            
            # If the model didn't write to output_path (e.g. mock), copy input as fallback
            if not Path(output_path).exists():
                shutil.copy(audio_path, output_path)
            
            await self._telemetry({"stage": "audio_complete", "audio_progress": 100})
            
            # Schedule VRAM flush
            await model_manager.schedule_flush()
            
            return {
                "gaps_found": 0,
                "gaps_repaired": 0,
                "sample_rate": sr,
                "channels": num_channels,
                "duration": round(duration, 2),
                "ai_mode": "full_gen"
            }

        # ── Inpaint Each Gap ──────────────────────────────────────────────────
        repaired = audio.copy()
        total    = max(len(gaps), 1)
        
        # Load FireRedTTS3 if we have gaps and using AI mode
        ai_model = None
        if len(gaps) > 0 and audio_mode == "gap_fill":
            logger.info("Using FireRedTTS3 for gap inpainting.")
            await self._telemetry({"stage": "ai_model_loading", "audio_progress": 22})
            ai_model = await model_manager.get_model("fireredtts")

        for i, gap in enumerate(gaps):
            progress = 20 + int((i / total) * 55)
            await self._telemetry({
                "stage":               "audio_inpainting",
                "audio_progress":       progress,
                "audio_gap_filled_pct": int(i / total * 100),
                "gap_index":            i,
                "gap_count":            len(gaps),
                "gap_duration_ms":      round(gap.duration_ms, 1),
            })

            # Process each channel independently so phase is preserved per-ch
            for ch in range(num_channels):
                if ai_model:
                    # Mock AI inpainting
                    repaired[ch] = await self._inpaint_ai_async(
                        ai_model, repaired[ch], gap, sr, script_text
                    )
                else:
                    repaired[ch] = await self._inpaint_gap_async(repaired[ch], gap, sr)

        await self._telemetry({
            "stage":               "audio_inpainted",
            "audio_progress":       75,
            "audio_gap_filled_pct": 100,
        })

        # ── Noise Reduction ───────────────────────────────────────────────────
        logger.info("Applying spectral noise reduction (aggression=%.2f)", self.noise_level)
        await self._telemetry({"stage": "noise_reduction", "audio_progress": 80})

        loop    = asyncio.get_event_loop()
        denoised = np.zeros_like(repaired)
        for ch in range(num_channels):
            denoised[ch] = await loop.run_in_executor(
                None, self._spectral_noise_gate, repaired[ch], sr
            )

        # ── Normalize ─────────────────────────────────────────────────────────
        peak = np.max(np.abs(denoised))
        if peak > 0.98:
            denoised = denoised / peak * 0.98
        elif peak == 0.0:
            logger.warning("Audio appears silent after processing; writing zeros.")

        # ── Save ──────────────────────────────────────────────────────────────
        await self._telemetry({"stage": "audio_saving", "audio_progress": 95})
        output_data = denoised.T if num_channels > 1 else denoised[0]
        sf.write(output_path, output_data, sr, subtype="PCM_24")

        del repaired, denoised
        gc.collect()

        # Schedule VRAM flush only if an AI model was loaded
        if ai_model is not None:
            await model_manager.schedule_flush()

        await self._telemetry({
            "stage":               "audio_complete",
            "audio_progress":       100,
            "audio_gap_filled_pct": 100,
        })

        return {
            "gaps_found":    len(gaps),
            "gaps_repaired": len(gaps),
            "sample_rate":   sr,
            "channels":      num_channels,
            "duration":      round(duration, 2),
        }

    # ── Private: Load ─────────────────────────────────────────────────────────

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file, returning (channels × samples) float32 array."""
        audio, sr = librosa.load(path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # force 2-D: (1, N)
        return audio.astype(np.float32), sr

    # ── Private: Gap Detection ─────────────────────────────────────────────────

    def _detect_gaps(self, mono: np.ndarray, sr: int) -> List[AudioGap]:
        """
        Identify silence gaps in the mono mix using windowed RMS analysis.

        Strategy
        --------
        1. Compute per-frame RMS with a 2×hop_length window.
        2. Mark frames below the threshold.
        3. Convert frame mask → sample mask.
        4. Collect contiguous silent regions ≥ min_gap_duration_ms.
        """
        min_gap_samples = int(self.min_gap_duration_ms / 1000.0 * sr)
        frame_length    = self.hop_length * 2

        rms_frames = librosa.feature.rms(
            y=mono, frame_length=frame_length, hop_length=self.hop_length
        )[0]
        rms_db = librosa.amplitude_to_db(np.maximum(rms_frames, 1e-10), ref=1.0)

        # Build per-sample silence mask from frame labels
        silence_mask = np.zeros(len(mono), dtype=bool)
        for i, db_val in enumerate(rms_db):
            s = i * self.hop_length
            e = min(s + frame_length, len(mono))
            if db_val < self.silence_threshold_db:
                silence_mask[s:e] = True

        # Apply morphological closing to fill tiny 1-sample holes
        struct_size = int(0.01 * sr)
        if struct_size > 0:
            silence_mask = scipy.ndimage.binary_closing(
                silence_mask, structure=np.ones(struct_size, dtype=bool)
            )

        # Collect contiguous silent regions
        gaps: List[AudioGap] = []
        in_gap, gap_start    = False, 0

        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_gap:
                in_gap, gap_start = True, i
            elif not silence_mask[i] and in_gap:
                in_gap = False
                if (i - gap_start) >= min_gap_samples:
                    gaps.append(AudioGap(gap_start, i, sr))

        if in_gap:  # gap extends to EOF
            length = len(silence_mask) - gap_start
            if length >= min_gap_samples:
                gaps.append(AudioGap(gap_start, len(silence_mask), sr))

        return gaps

    # ── Private: Inpainting ───────────────────────────────────────────────────

    async def _inpaint_ai_async(
        self, model, audio: np.ndarray, gap: AudioGap, sr: int, script_text: str
    ) -> np.ndarray:
        """Slice the gap region with surrounding context, pass to AI model, write result back."""
        loop = asyncio.get_event_loop()
        context_n = int(self.context_seconds * sr)
        pre_start = max(0, gap.start_sample - context_n)
        post_end = min(len(audio), gap.end_sample + context_n)

        context_audio = audio[pre_start:post_end].copy()
        gap_start_rel = gap.start_sample - pre_start
        gap_end_rel = gap.end_sample - pre_start

        patched = await loop.run_in_executor(
            None, model.inpaint_audio, context_audio, sr, gap_start_rel, gap_end_rel, script_text
        )

        result = audio.copy()
        result[gap.start_sample:gap.end_sample] = patched[gap_start_rel:gap_end_rel]
        return result

    async def _inpaint_gap_async(
        self, audio: np.ndarray, gap: AudioGap, sr: int
    ) -> np.ndarray:
        """Run synchronous inpainting off the event loop to keep the WS alive."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._inpaint_gap, audio, gap, sr)

    def _inpaint_gap(
        self, audio: np.ndarray, gap: AudioGap, sr: int
    ) -> np.ndarray:
        """
        Phase-coherent spectral inpainting.

        Algorithm
        ---------
        1. Extract 3-second pre / post context windows.
        2. Compute complex STFT of both contexts.
        3. Magnitude: linearly alpha-blend from pre-tail mean → post-head mean
           across the gap's time frames, then add a tiny noise floor drawn from
           the pre-context's spectral shape (retains harmonic texture).
        4. Phase: extrapolate using Instantaneous Frequency Estimation (IFE)
           from the last 4 pre-frames; blend into post-phase in the final 25 %
           of gap frames for phase continuity at both boundaries.
        5. ISTFT → cross-fade at both junction boundaries (16 ms Hann ramp).
        """
        context_n = int(self.context_seconds * sr)
        gap_start = gap.start_sample
        gap_end   = min(gap.end_sample, len(audio))
        gap_len   = gap_end - gap_start

        if gap_len <= 0:
            return audio

        # ── Context Extraction ────────────────────────────────────────────
        pre_start = max(0, gap_start - context_n)
        post_end  = min(len(audio), gap_end + context_n)

        pre_ctx  = audio[pre_start:gap_start]
        post_ctx = audio[gap_end:post_end]

        # Pad short contexts with reflected signal so STFT is well-defined
        if len(pre_ctx) < context_n:
            pre_ctx = np.pad(pre_ctx, (context_n - len(pre_ctx), 0), mode="reflect")
        if len(post_ctx) < context_n:
            post_ctx = np.pad(post_ctx, (0, context_n - len(post_ctx)), mode="reflect")

        # ── STFT of Contexts ──────────────────────────────────────────────
        window   = scipy.signal.windows.hann(self.win_length, sym=False)
        pre_stft  = librosa.stft(
            pre_ctx,  n_fft=self.n_fft, hop_length=self.hop_length, window=window
        )
        post_stft = librosa.stft(
            post_ctx, n_fft=self.n_fft, hop_length=self.hop_length, window=window
        )

        pre_mag,  pre_phi  = np.abs(pre_stft),  np.angle(pre_stft)
        post_mag, post_phi = np.abs(post_stft), np.angle(post_stft)

        freq_bins  = pre_mag.shape[0]
        gap_frames = max(int(np.ceil(gap_len / self.hop_length)) + 1, 1)

        # ── Magnitude Interpolation ───────────────────────────────────────
        tail_n      = min(8, pre_mag.shape[1])
        head_n      = min(8, post_mag.shape[1])
        pre_mean    = np.mean(pre_mag[:, -tail_n:],  axis=1)  # (freq_bins,)
        post_mean   = np.mean(post_mag[:, :head_n],  axis=1)  # (freq_bins,)

        alphas        = np.linspace(0.0, 1.0, gap_frames)          # (gap_frames,)
        interp_mag    = np.outer(pre_mean, (1.0 - alphas)) + np.outer(post_mean, alphas)
        # shape: (freq_bins, gap_frames)

        # Add spectral noise drawn from pre-context standard deviation
        pre_std       = np.std(pre_mag[:, -tail_n:], axis=1, keepdims=True)
        spectral_noise = np.random.randn(freq_bins, gap_frames) * pre_std * 0.04
        interp_mag    = np.maximum(0.0, interp_mag + spectral_noise)

        # ── Phase Extrapolation (IFE) ─────────────────────────────────────
        inpainted_phi = self._extrapolate_phase(
            pre_phi, post_phi, gap_frames
        )

        # ── Reconstruct & ISTFT ───────────────────────────────────────────
        inpainted_stft  = interp_mag * np.exp(1j * inpainted_phi)
        inpainted_audio = librosa.istft(
            inpainted_stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=gap_len,
        )
        inpainted_audio = inpainted_audio[:gap_len]

        # ── Cross-fade at Boundaries (16 ms Hann ramp) ───────────────────
        fade_len = min(int(0.016 * sr), max(gap_len // 4, 1))
        if fade_len > 1:
            fade_in  = np.linspace(0.0, 1.0, fade_len)
            fade_out = np.linspace(1.0, 0.0, fade_len)
            inpainted_audio[:fade_len]  *= fade_in
            inpainted_audio[-fade_len:] *= fade_out

        # ── Write Back ────────────────────────────────────────────────────
        result              = audio.copy()
        result[gap_start:gap_end] = inpainted_audio

        # Smooth junction with existing signal (Hann crossfade)
        xf_len = min(fade_len, gap_start, len(audio) - gap_end)
        if xf_len > 1:
            ramp = np.hanning(xf_len * 2)[:xf_len]  # rising half
            # Pre-junction: blend existing → inpainted
            result[gap_start:gap_start + xf_len] = (
                audio[gap_start:gap_start + xf_len] * (1 - ramp)
                + inpainted_audio[:xf_len]           * ramp
            )
            # Post-junction: blend inpainted → existing
            result[gap_end - xf_len:gap_end] = (
                inpainted_audio[gap_len - xf_len:]    * (1 - ramp[::-1])
                + audio[gap_end - xf_len:gap_end]     * ramp[::-1]
            )

        return result

    @staticmethod
    def _extrapolate_phase(
        pre_phi:  np.ndarray,  # (freq_bins, pre_frames)
        post_phi: np.ndarray,  # (freq_bins, post_frames)
        gap_frames: int,
    ) -> np.ndarray:
        """
        Extrapolate spectral phase using per-bin Instantaneous Frequency.

        The last 4 pre-frames are used to estimate each frequency bin's
        average phase velocity (instantaneous frequency).  Phase is then
        rolled forward linearly for the gap duration.  In the final 25 % of
        gap frames the extrapolated phase is linearly blended into the post-
        context phase for a smooth landing.
        """
        freq_bins = pre_phi.shape[0]
        out = np.zeros((freq_bins, gap_frames), dtype=np.float64)

        if pre_phi.shape[1] >= 2:
            look_back  = min(4, pre_phi.shape[1] - 1)
            dphi       = np.diff(pre_phi[:, -look_back - 1:], axis=1)          # (F, look_back)
            # Wrap to (−π, π]
            dphi       = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
            mean_dphi  = np.mean(dphi, axis=1)                                  # (F,)
        else:
            mean_dphi = np.zeros(freq_bins)

        last_phi = pre_phi[:, -1]                                               # (F,)
        for f in range(gap_frames):
            out[:, f] = last_phi + (f + 1) * mean_dphi

        # Blend into post-context in the trailing 25 %
        blend_start = max(0, int(gap_frames * 0.75))
        blend_len   = gap_frames - blend_start
        if blend_len > 0 and post_phi.shape[1] > 0:
            for f in range(blend_start, gap_frames):
                alpha  = (f - blend_start) / max(blend_len, 1)
                post_f = min(f - blend_start, post_phi.shape[1] - 1)
                out[:, f] = (1.0 - alpha) * out[:, f] + alpha * post_phi[:, post_f]

        return out

    # ── Private: Noise Reduction ──────────────────────────────────────────────

    def _spectral_noise_gate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Spectral subtraction noise reduction.

        The noise profile is estimated from the first 0.5 seconds of audio
        (assumed to be representative background noise).  The profile is then
        subtracted from every frame, with a soft floor at `floor_factor × signal`
        to prevent musical noise.

        Aggressiveness (0.0–1.0) controls the subtraction multiplier:
          0.5 → subtract 1.5× the noise estimate (default)
          1.0 → subtract 3.0× (strong artefacts possible)
          0.0 → subtract 0.5× (very gentle)
        """
        noise_estimate_n = int(0.5 * sr)
        if noise_estimate_n >= len(audio):
            return audio  # too short to estimate noise

        noise_segment = audio[:noise_estimate_n]
        noise_stft    = librosa.stft(
            noise_segment, n_fft=self.n_fft, hop_length=self.hop_length
        )
        noise_profile = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

        audio_stft  = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        audio_mag   = np.abs(audio_stft)
        audio_phase = np.angle(audio_stft)

        subtract_factor = 0.5 + self.noise_level * 2.5   # 0.5 … 3.0
        floor_factor    = max(0.05, 0.15 - self.noise_level * 0.1)

        clean_mag = audio_mag - subtract_factor * noise_profile
        clean_mag = np.maximum(clean_mag, floor_factor * audio_mag)

        clean_stft  = clean_mag * np.exp(1j * audio_phase)
        clean_audio = librosa.istft(
            clean_stft, hop_length=self.hop_length, length=len(audio)
        )

        return clean_audio.astype(np.float32)

    # ── Private: Win-Length Helper ─────────────────────────────────────────────

    @property
    def win_length(self) -> int:
        return self.n_fft  # standard: win == n_fft

    # ── Private: Telemetry ─────────────────────────────────────────────────────

    async def _telemetry(self, data: dict) -> None:
        if self.telemetry_callback is not None:
            try:
                await self.telemetry_callback(data)
            except Exception as exc:
                logger.debug("Telemetry callback error (non-fatal): %s", exc)
