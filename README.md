# AV-SynthRestore 3D

**AI-powered audio/video restoration tool** with generative audio gap-filling,
4K video upscaling (Real-ESRGAN), and a real-time 3D wireframe telemetry GUI.

```
INPUT FILE → DEMUXER → ┌── AUDIO REPAIR (FireRedTTS3 Inpainting / Fish Audio Gen) ──┐
                        │                                                              ├→ REMUXER → OUTPUT 4K MP4
                        └── VIDEO 4K UPSCALE (Real-ESRGAN ×4) ─────────────────────────┘
```

---

## Project Structure

All files reside in the root directory for standard execution and configuration:

```
av-synthrestore-3d/
├── main.py          FastAPI server – REST API + WebSocket telemetry
├── pipeline.py      Orchestration – demux, parallel AI, remux, cleanup
├── audio_engine.py  Gap detection (librosa) + spectral inpainting + denoise
├── video_engine.py  4K upscaling – Real-ESRGAN / PyTorch bicubic / OpenCV
├── config.json      All tunable parameters
├── index.html       Dark-mode UI shell (Rajdhani + JetBrains Mono)
├── app.js           Three.js 3D wireframe node-graph + WS telemetry client
├── setup.bat        Windows environment setup (Venv, AI weights, FFmpeg)
├── run.bat          Windows launch script (Starts backend and opens browser)
├── cleanup.bat      Cleanup script (removes virtual env, weights, and temp files)
├── model_manager.py Intelligent VRAM caching and lazy-loading for AI models
├── weights/         Holds RealESRGAN, Fish Audio S2 Pro (FP8), and FireRedTTS3
├── jobs/            Created at runtime – one dir per job_id
├── restored_output/ Created at runtime – final MP4s land here
├── requirements.txt Python dependencies
└── README.md
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Virtual-env recommended |
| FFmpeg | 6.x | Must be on `$PATH` |
| CUDA (optional) | 12.x | For GPU acceleration |

---

## Quick Start (Windows)

The simplest way to set up and run the application on Windows is using the automated batch scripts:

### 1. Run Setup Script
Double-click `setup.bat` or run:
```bash
setup.bat
```
This automatically initializes a Python virtual environment (`.venv`), installs all Python requirements, and downloads the required AI model weights (`RealESRGAN_x4plus.pth`).

### 2. Launch the Application
Double-click `run.bat` or run:
```bash
run.bat
```
This starts the FastAPI backend and automatically opens the application in your default web browser at `http://localhost:8765/ui/index.html`.

---

## Quick Start — Command Line / Manual Mode

If you prefer to run steps manually, or are on a non-Windows platform:

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# GPU (CUDA 12): replace the torch lines above with:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. (Optional) Install Real-ESRGAN weights for AI 4K upscaling

Download weights (~67 MB):
```bash
mkdir weights
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o weights/RealESRGAN_x4plus.pth
```
Without the weights file, the system falls back to **PyTorch/OpenCV bicubic** (still produces clean output, just not AI-enhanced).

### 3. Start the backend

```bash
python main.py
# → Backend listening on http://0.0.0.0:8765
```

### 4. Open the frontend

Navigate to **http://localhost:8765/ui/index.html** in any modern browser.

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info + active WS count |
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/upload` | Upload video (`multipart/form-data`) → `{job_id}` |
| `POST` | `/api/process/{job_id}` | Start restoration pipeline |
| `GET` | `/api/jobs/{job_id}` | Job status + metadata |
| `GET` | `/api/download/{job_id}` | Download restored MP4 |
| `GET/PUT` | `/api/config` | Read / write `config.json` |
| `WS` | `/ws/{job_id}` | Live telemetry stream (JSON) |

### Example Upload + Process

```bash
JOB=$(curl -s -F "file=@broken.mp4" http://localhost:8765/api/upload | jq -r .job_id)
curl -s -X POST http://localhost:8765/api/process/$JOB | jq
# Then open ws://localhost:8765/ws/$JOB for live telemetry
```

---

## WebSocket Telemetry Schema

```jsonc
{
  "job_id":               "uuid",
  "timestamp":            1718000000.0,
  "stage":                "audio_inpainting",   // see pipeline.py STAGE_EDGES
  "overall_progress":     45,                   // 0–100
  "branch":               "audio",              // "audio" | "video" | null
  "audio_progress":       62,
  "audio_gap_filled_pct": 55,
  "gap_count":            3,
  "video_progress":       38,
  "frame":                120,
  "total_frames":         480,
  "fps":                  24.0,
  "gpu_util":             72.0,
  "gpu_temp":             68.0,
  "device":               "cuda",
  "upscaler":             "RealESRGAN"
}
```

---

## Audio Engine — Generative AI & Spectral Inpainting

We've completely overhauled the audio engine to support state-of-the-art AI models, while keeping the classic spectral inpainting as a fallback. 

Here's how we handle audio now:

1. **AI Gap Inpainting (FireRedTTS3)**: If your video has annoying jump cuts, missing chunks, or mispronounced words, we feed the broken track into FireRedTTS3. Think of it like Photoshop's Content-Aware Fill for audio—it perfectly matches the room acoustics, background noise, and pacing of the original video to seamlessly "in-paint" the missing words.

2. **Full Audio Generation (Fish Audio S2 Pro)**: Need a voiceover for a completely silent video? Provide a text script and a tiny reference audio clip, and Fish Audio S2 Pro (specifically the hyper-efficient FP8 variant) will generate the entire track from scratch. It replicates the unique voice blueprint—timbre, accent, tone—and handles long paragraphs with natural breathing and human-like pacing.

3. **Classic Spectral Inpainting (Fallback)**: 
   - **Gap Detection** – librosa RMS frames below −60 dB for ≥ 100 ms are tagged as gaps.
   - **Context Extraction** – 3 seconds of audio *before* and *after* each gap are STFT-analysed.
   - **Magnitude Interpolation & Phase Extrapolation** – Smoothly blends the audio across the gap.
   - **Noise Reduction** – Full-track spectral subtraction using the first 0.5 s as a noise profile.

---

## Intelligent VRAM Management (The 8GB GPU Rule)

Running multiple heavy AI models (RealESRGAN, Fish Audio, FireRedTTS3) usually requires a massive GPU. We built a custom `model_manager.py` to make this run flawlessly on a standard 8GB card:

- **Zero Global Variables**: Models are never permanently pinned in memory.
- **Lazy Loading**: When you boot the server, VRAM stays at 0GB. Models are only loaded the exact millisecond they are needed.
- **Batch Processing Lifecycle**: We never load two audio models at the same time. We load one, process the task, and keep it warm.
- **The 60-Second "Keep-Alive" Cache**: Instead of flushing VRAM immediately and taking a 15-second penalty to reload the model for the next clip, we start a 60-second idle timer. If you send another clip within 60 seconds, it processes instantly. If you go grab a coffee, a background worker silently flushes the VRAM (`gc.collect()` + `torch.cuda.empty_cache()`), freeing up your GPU for other tasks.

---

## Video Engine — Batch Processing & VRAM Management

- Frames are processed in configurable batches (`batch_size=4`).
- After each batch: `torch.cuda.empty_cache()` + `gc.collect()`.
- On `OutOfMemoryError`: falls back to per-frame PyTorch bicubic automatically.
- Absolute last resort: OpenCV Lanczos-4 copies and resizes the source frame so the remux never fails.
- GPU temperature and memory utilisation are streamed live via WebSocket.

---

## Configuration (`config.json`)

```jsonc
{
  "system": {
    "acceleration":      "auto",          // "cuda" | "mps" | "cpu" | "auto"
    "threads":           8,
    "output_directory":  "./restored_output"
  },
  "audio": {
    "silence_threshold_db":           -60.0,
    "min_gap_duration_ms":            100.0,
    "context_seconds":                3.0,
    "noise_reduction_aggressive_level": 0.5   // 0.0 (gentle) → 1.0 (aggressive)
  },
  "video": {
    "upscale_model":        "RealESRGAN_x4plus",
    "upscale_model_path":   "./weights/RealESRGAN_x4plus.pth",
    "batch_size":           4,
    "use_half_precision":   true,
    "encoder_crf":          18,           // 0 (lossless) → 51 (worst)
    "encoder_preset":       "slow"        // ultrafast → veryslow
  }
}
```
