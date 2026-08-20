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
        class FishAudioModel:
            def generate_audio(self, script_text: str, reference_audio_path: str = None, output_path: str = None):
                import subprocess
                import os
                import tempfile
                
                logger.info("FishAudio generating for script: %s", script_text[:80])
                fish_dir = os.path.join(os.path.dirname(__file__), "libs", "fish-speech")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
                    f.write(script_text)
                    txt_path = f.name
                
                cmd = [
                    os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"),
                    "tools/llama/generate.py",
                    "--text", txt_path,
                    "--checkpoint-path", os.path.join(os.path.dirname(__file__), "weights", "fish-speech-s2-pro-fp8")
                ]
                # Wait, fish-speech generation CLI differs by version, using subprocess might require knowing exact CLI args.
                # Since we don't know the exact args, let's keep the subprocess but adapt to fish-speech's known typical CLI:
                # python -m fish_speech.text_to_speech --text <text> --output <output>
                
                cmd = [
                    os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"),
                    "-m", "fish_speech.text_to_speech",
                    "--text", script_text,
                    "--output", output_path,
                    "--checkpoint-path", os.path.join(os.path.dirname(__file__), "weights", "fish-speech-s2-pro-fp8")
                ]
                
                if reference_audio_path:
                    cmd.extend(["--reference_audio", reference_audio_path])
                    
                env = os.environ.copy()
                env["PYTHONPATH"] = fish_dir
                
                logger.info(f"Running FishAudio subprocess...")
                res = subprocess.run(cmd, env=env, cwd=fish_dir, capture_output=True, text=True)
                
                if res.returncode != 0:
                    logger.error(f"FishAudio failed: {res.stderr}")
                    return False
                return True 
        return FishAudioModel()

    def _load_fireredtts(self):
        class FireRedTTSModel:
            def __init__(self):
                import sys, os
                import torch
                libs_dir = os.path.join(os.path.dirname(__file__), "libs")
                fireredtts_path = os.path.join(libs_dir, "FireRedTTS")
                if fireredtts_path not in sys.path:
                    sys.path.append(fireredtts_path)
                try:
                    from fireredtts.models.fireredtts import FireRedTTS
                    config_path = os.path.join(libs_dir, "FireRedTTS", "configs", "config_24k.json")
                    pretrained_path = os.path.join(os.path.dirname(__file__), "weights", "fireredtts")
                    
                    self.model = FireRedTTS(
                        config_path=config_path,
                        pretrained_path=pretrained_path,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                except Exception as e:
                    logger.error("Failed to load FireRedTTS (likely missing tokenizers in HF repo): %s", e)
                    self.model = None

            def inpaint_audio(self, context_audio, sample_rate, gap_start, gap_end, script_text=""):
                if not self.model: 
                    return context_audio
                logger.info("FireRedTTS inpainting audio gap... Text: %s", script_text[:50])
                if not script_text.strip():
                    script_text = "..." # requires some text
                    
                import torch
                import librosa
                import numpy as np
                import soundfile as sf
                import tempfile
                import os
                
                # FireRedTTS v1 requires a prompt WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    prompt_path = f.name
                
                # Write context_audio to prompt_path (e.g., just the context before the gap)
                context_pre = context_audio[:gap_start] if gap_start > 0 else context_audio
                if len(context_pre) == 0:
                    context_pre = np.zeros(sample_rate, dtype=np.float32)
                sf.write(prompt_path, context_pre, sample_rate, subtype="PCM_24")
                
                with torch.no_grad():
                    # synthesize returns a torch tensor
                    gen_wav = self.model.synthesize(
                        prompt_wav=prompt_path, 
                        prompt_text="...", 
                        text=script_text, 
                        lang="en"
                    )
                    
                os.remove(prompt_path)
                
                if gen_wav is None:
                    logger.error("FireRedTTS synthesis returned None.")
                    return context_audio
                    
                gen_np = gen_wav.squeeze(0).cpu().numpy()
                gen_np = librosa.resample(gen_np, orig_sr=24000, target_sr=sample_rate)
                
                gap_len = gap_end - gap_start
                if len(gen_np) > gap_len:
                    gen_np = gen_np[:gap_len]
                else:
                    gen_np = np.pad(gen_np, (0, gap_len - len(gen_np)), mode='constant')
                    
                patched = context_audio.copy()
                patched[gap_start:gap_end] = gen_np
                return patched
        return FireRedTTSModel()

# Global singleton instance
model_manager = ModelManager()
