# AV-SynthRestore 3D

**AI-powered audio/video restoration tool** with generative audio gap-filling,
4K video upscaling (Real-ESRGAN), and a real-time 3D wireframe telemetry GUI.

```
INPUT FILE → DEMUXER → ┌── AUDIO REPAIR (spectral inpainting) ──┐
                        │                                         ├→ REMUXER → OUTPUT 4K MP4
                        └── VIDEO 4K UPSCALE (Real-ESRGAN ×4) ──┘
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
├── setup.bat        Windows environment setup (Venv, weights)
├── run.bat          Windows launch script (Starts backend and opens browser)
├── cleanup.bat      Cleanup script (removes virtual env, weights, and temp files)
├── weights/         Created at runtime or setup – holds RealESRGAN_x4plus.pth
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

## Audio Engine — How Spectral Inpainting Works

1. **Gap Detection** – librosa RMS frames below −60 dB for ≥ 100 ms are tagged as gaps. A morphological closing pass merges near-adjacent micro-silences.

2. **Context Extraction** – 3 seconds of audio *before* and *after* each gap are STFT-analysed (`n_fft=2048`, `hop=512`).

3. **Magnitude Interpolation** – The mean spectral magnitude of the last 8 pre-frames is linearly blended (α from 0→1) into the mean of the first 8 post-frames across the gap. A small noise term drawn from pre-context standard deviation preserves harmonic texture.

4. **Phase Extrapolation (IFE)** – Per-bin instantaneous frequency is estimated from the last 4 pre-frames and rolled forward. The trailing 25 % of gap frames blend into post-context phase for a smooth landing.

5. **Cross-fading** – 16 ms Hann ramps are applied at both boundary junctions to prevent clicks.

6. **Noise Reduction** – Full-track spectral subtraction using the first 0.5 s as a noise profile, with a soft floor at `0.1 × signal` to prevent musical noise artefacts.

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
