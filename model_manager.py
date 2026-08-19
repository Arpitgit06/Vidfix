"""
model_manager.py
Manages loading, caching, and VRAM flushing for AI Audio Models.
Implements the 60-second Keep-Alive rule and lazy loading.
"""

import asyncio
import gc
import logging
import time

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._fish_audio = None
        self._firered_tts = None
        self._active_model_name = None
        self._idle_timer_task = None
        # Lazy lock — created on first use inside the running event loop
        self._lock = None

    def _get_lock(self):
        """Create the asyncio.Lock lazily so it's bound to the correct event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_model(self, model_name: str):
        """
        Lazy loads the requested model.
        Flushes any currently loaded DIFFERENT model to ensure VRAM limits (8GB).
        Cancels any pending idle flush timer.
        """
        async with self._get_lock():
            if self._idle_timer_task is not None:
                self._idle_timer_task.cancel()
                self._idle_timer_task = None
                logger.info("Cancelled idle flush timer because %s was requested.", model_name)

            if self._active_model_name and self._active_model_name != model_name:
                logger.info("Different model (%s) is currently loaded. Flushing VRAM first...", self._active_model_name)
                self._flush_vram_sync()

            if model_name == "fish_audio":
                if self._fish_audio is None:
                    logger.info("Loading Fish Audio S2 Pro (FP8) into VRAM...")
                    start_time = time.time()
                    self._fish_audio = self._load_fish_audio()
                    logger.info("Fish Audio loaded in %.2fs", time.time() - start_time)
                self._active_model_name = "fish_audio"
                return self._fish_audio

            elif model_name == "fireredtts":
                if self._firered_tts is None:
                    logger.info("Loading FireRedTTS3 into VRAM...")
                    start_time = time.time()
                    self._firered_tts = self._load_fireredtts()
                    logger.info("FireRedTTS3 loaded in %.2fs", time.time() - start_time)
                self._active_model_name = "fireredtts"
                return self._firered_tts
            
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

    async def schedule_flush(self):
        """
        Starts a 60-second countdown. If no new requests arrive, VRAM is flushed.
        """
        async with self._get_lock():
            if self._idle_timer_task is not None:
                self._idle_timer_task.cancel()
            
            logger.info("Scheduling VRAM flush in 60 seconds...")
            self._idle_timer_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        """Wait 60s then flush. Acquires lock separately to avoid deadlock."""
        try:
            await asyncio.sleep(60)
            await self._execute_flush()
        except asyncio.CancelledError:
            pass  # Timer was cancelled by a new request

    async def _execute_flush(self):
        """Acquire lock and flush VRAM — called by the delayed timer."""
        async with self._get_lock():
            logger.info("60 seconds idle timeout reached. Flushing VRAM...")
            self._flush_vram_sync()
            self._idle_timer_task = None

    def _flush_vram_sync(self):
        """
        Synchronous VRAM flush. Deletes model references and forces PyTorch garbage collection.
        """
        if self._fish_audio is not None:
            del self._fish_audio
            self._fish_audio = None
        
        if self._firered_tts is not None:
            del self._firered_tts
            self._firered_tts = None
        
        self._active_model_name = None
        
        # Force garbage collection
        gc.collect()
        
        # Empty PyTorch CUDA cache
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("VRAM has been flushed completely.")

    # ── Model Mock/Stubs (Replace with actual initialization code) ─────────────
    
    def _load_fish_audio(self):
        # TODO: Initialize Fish Audio S2 Pro FP8 from weights\\fish-speech-s2-pro-fp8
        # Since this is a placeholder/mock object for now, we will return a dummy class
        # that handles the interface. In production, this would be:
        # from transformers import ...
        class FishAudioMock:
            def generate_audio(self, script_text: str, reference_audio_path: str = None, output_path: str = None):
                logger.info("FishAudio generating for script: %s", script_text[:80])
                # Simulate generation time
                time.sleep(2)
                # In real scenario, generate audio and write to output_path
                # For mock, write a silent WAV if output_path is provided
                if output_path:
                    import numpy as np
                    import soundfile as sf
                    # Generate 5 seconds of silence as placeholder
                    sr = 48000
                    silence = np.zeros(sr * 5, dtype=np.float32)
                    sf.write(output_path, silence, sr, subtype="PCM_24")
                return True 
        return FishAudioMock()

    def _load_fireredtts(self):
        # TODO: Initialize FireRedTTS3 from weights\\fireredtts
        class FireRedTTSMock:
            def inpaint_audio(self, context_audio, sample_rate, gap_start, gap_end):
                logger.info("FireRedTTS3 inpainting audio gap (samples %d-%d)...", gap_start, gap_end)
                time.sleep(1)
                # In real scenario, return patched numpy array with the gap filled
                # For mock, just return the input unchanged
                return context_audio 
        return FireRedTTSMock()

# Global singleton instance
model_manager = ModelManager()
