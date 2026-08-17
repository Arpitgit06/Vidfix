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
        self._lock = asyncio.Lock()

    async def get_model(self, model_name: str):
        """
        Lazy loads the requested model.
        Flushes any currently loaded DIFFERENT model to ensure VRAM limits (8GB).
        Cancels any pending idle flush timer.
        """
        async with self._lock:
            if self._idle_timer_task is not None:
                self._idle_timer_task.cancel()
                self._idle_timer_task = None
                logger.info(f"Cancelled idle flush timer because {model_name} was requested.")

            if self._active_model_name and self._active_model_name != model_name:
                logger.info(f"Different model ({self._active_model_name}) is currently loaded. Flushing VRAM first...")
                self._flush_vram_sync()

            if model_name == "fish_audio":
                if self._fish_audio is None:
                    logger.info("Loading Fish Audio S2 Pro (FP8) into VRAM...")
                    start_time = time.time()
                    self._fish_audio = self._load_fish_audio()
                    logger.info(f"Fish Audio loaded in {time.time() - start_time:.2f}s")
                self._active_model_name = "fish_audio"
                return self._fish_audio

            elif model_name == "fireredtts":
                if self._firered_tts is None:
                    logger.info("Loading FireRedTTS3 into VRAM...")
                    start_time = time.time()
                    self._firered_tts = self._load_fireredtts()
                    logger.info(f"FireRedTTS3 loaded in {time.time() - start_time:.2f}s")
                self._active_model_name = "fireredtts"
                return self._firered_tts
            
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

    async def schedule_flush(self):
        """
        Starts a 60-second countdown. If no new requests arrive, VRAM is flushed.
        """
        async with self._lock:
            if self._idle_timer_task is not None:
                self._idle_timer_task.cancel()
            
            logger.info("Scheduling VRAM flush in 60 seconds...")
            self._idle_timer_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        try:
            await asyncio.sleep(60)
            async with self._lock:
                logger.info("60 seconds idle timeout reached. Flushing VRAM...")
                self._flush_vram_sync()
        except asyncio.CancelledError:
            pass # Timer was cancelled by a new request

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
            def generate_audio(self, script_text: str, reference_audio_path: str = None):
                logger.info(f"FishAudio generating for script: {script_text}")
                # Simulate generation time
                time.sleep(2)
                # In real scenario, return generated numpy array or save to file
                return True 
        return FishAudioMock()

    def _load_fireredtts(self):
        # TODO: Initialize FireRedTTS3 from weights\\fireredtts
        class FireRedTTSMock:
            def inpaint_audio(self, broken_audio, sample_rate):
                logger.info("FireRedTTS3 inpainting audio gap...")
                time.sleep(1)
                # In real scenario, return patched numpy array
                return broken_audio 
        return FireRedTTSMock()

# Global singleton instance
model_manager = ModelManager()
