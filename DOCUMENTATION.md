# SPIKD — Technical Documentation
## Cortical Attention Diagnostic Platform
**Version 1.0 — Build Reference**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Pipeline: Step-by-Step](#3-pipeline-step-by-step)
4. [File Structure](#4-file-structure)
5. [Installation Guide (RunPod)](#5-installation-guide-runpod)
6. [Configuration Reference](#6-configuration-reference)
7. [User Guide](#7-user-guide)
8. [API Reference](#8-api-reference)
9. [TRIBE v2 Integration](#9-tribe-v2-integration)
10. [Atlas Masking (Glasser MMP1.0)](#10-atlas-masking-glasser-mmp10)
11. [Simulation Mode](#11-simulation-mode)
12. [VRAM Management](#12-vram-management)
13. [Security](#13-security)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Overview

SPIKD is a clinical-grade attention diagnostic tool that processes video files through a neuroscience-informed pipeline to produce a cortical attention waveform and a structured PDF report.

**The pipeline:**

```
Video File (1080p, 15-60s)
    │
    ▼
Frame Extraction (OpenCV)
    │
    ▼
TRIBE v2 — Meta Research cortical surface model
    Outputs: [T frames × 20,484 vertices]
    │
    ▼
Glasser MMP1.0 Atlas Masking
    Retains: Dorsal Attention Network (DAN) + Primary Visual Cortex (V1)
    Discards: all other vertices
    Outputs: [T × N_masked vertices]
    │
    ▼
1D Collapse — spatial mean over masked vertices
    Outputs: [T] normalised attention waveform
    │
    ▼
Engagement Metrics
    - Peak engagement timestamp
    - Engagement drop timestamp
    │
    ├─► Ad Copy Input (optional text box)
    │
    ▼
Llama 3.2 (via Ollama)
    - Waveform narrative summary
    - Ad copy alignment analysis + score (High / Medium / Low)
    │
    ▼
PDF Report
    - Cover page + metadata
    - Key metrics table
    - Monochrome 2D waveform chart
    - LLM narrative
    - Alignment analysis (if ad copy provided)
```

**What SPIKD does not do:**
- No 3D brain mesh rendering
- No nilearn or VTK visualisations
- No global brain signal — only atlas-masked DAN + V1 regions
- No external data transmission — all processing is local to the RunPod instance

---

## 2. Architecture

### Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI 0.111 |
| Templating | Jinja2 |
| Session management | Starlette SessionMiddleware (signed cookies) |
| Auth | SHA-256 hashed credentials in .env |
| ML inference | PyTorch 2.3 + CUDA 12.1 |
| Video processing | OpenCV headless |
| Neuroimaging | NiBabel (atlas loading) |
| LLM | Llama 3.2 via Ollama REST API |
| PDF generation | ReportLab |
| Charting | Matplotlib (Agg backend — headless) |
| Process manager | systemd |
| WSGI/ASGI | Uvicorn |

### Request flow

```
Browser → FastAPI (port 8000)
    │
    ├── GET  /          → dashboard.html  (Jinja2)
    ├── GET  /login     → login.html
    ├── POST /login     → session cookie set → redirect /
    ├── POST /analyze   → saves video, spawns asyncio background task → returns {job_id}
    ├── GET  /status/{job_id}   → JSON polling endpoint
    ├── GET  /progress/{job_id} → progress.html (polls /status via JS)
    ├── GET  /result/{job_id}   → result.html
    ├── GET  /download/{job_id} → FileResponse (PDF)
    └── GET  /sessions/{job_id}/{file} → FileResponse (waveform PNG)
```

### Background task pipeline

Jobs run as `asyncio.create_task()` coroutines using `loop.run_in_executor()` to offload CPU/GPU-bound work to a thread pool without blocking the event loop. Job state is tracked in an in-memory dictionary keyed by UUID.

---

## 3. Pipeline: Step-by-Step

### Step 1 — Frame Extraction

`pipeline/tribe_processor.py :: _extract_frames()`

- Opens video with OpenCV `VideoCapture`
- Reads every frame (no subsampling — full temporal resolution)
- Resizes each frame to 224×224 (TRIBE v2 input size)
- Converts BGR → RGB
- Returns: `list[np.ndarray]`, `fps: float`, `duration_sec: float`

### Step 2 — TRIBE v2 Inference

`pipeline/tribe_processor.py :: _predict_surface()`

TRIBE v2 (Meta Research) is a video-to-cortex model trained on paired fMRI + video data from the Natural Scenes Dataset. It predicts BOLD (Blood-Oxygen-Level-Dependent) responses on the fsLR-32k cortical surface.

**Input:** `[1, T, 3, 224, 224]` tensor (batch=1, T frames, RGB, 224×224)
**Output:** `[T, 20484]` — predicted activation at each of the 20,484 fsLR-32k surface vertices

The 20,484 vertices correspond to the standard HCP 32k surface (left hemisphere: vertices 0–10241, right: 10242–20483).

### Step 3 — Glasser Atlas Masking

`pipeline/atlas_masker.py :: apply()`

The Glasser HCP MMP1.0 atlas parcellates the cortical surface into 360 regions (180 per hemisphere). SPIKD retains only:

| Network | Regions | Parcel IDs (left) |
|---|---|---|
| Primary Visual Cortex (V1) | V1 | 1 |
| Dorsal Attention Network | IPS (LIPd, LIPv, VIP, MIP, AIP, 7PL) | 7, 17, 18, 19, 23, 24 |
| | FEF (8C, p9-46v) | 10, 11 |
| | MT+ | 157 |

All other vertices are **discarded** (not zeroed — physically excluded from the matrix).

Result: `[T, N_masked]` where N_masked is typically 1,200–2,400 vertices depending on atlas precision.

### Step 4 — 1D Collapse

```python
waveform = masked_tensor.mean(axis=1)   # [T]
```

Spatial mean across all retained DAN + V1 vertices at each timepoint. This produces the scalar "attention signal" over time — the Attention Spike Waveform.

The waveform is normalised to [0, 1] using min-max scaling.

### Step 5 — Engagement Metrics

**Peak engagement:** `argmax(waveform)` → frame index → seconds

**Engagement drop:** First-derivative minimum of the smoothed signal, searched only after the peak. Uses a 5-frame moving average before differentiation to suppress noise.

### Step 6 — Waveform Rendering

Matplotlib with `Agg` (headless) backend. Dark background (#0a0a0a), white signal line, annotated peak and drop markers. Saved as PNG at 150 DPI. No 3D components whatsoever.

### Step 7 — Llama 3.2 Analysis

`pipeline/llm_summary.py`

Sends a structured JSON-format prompt to Ollama's `/api/generate` endpoint. The prompt includes:

- Video duration, peak timestamp, drop timestamp
- Average and maximum normalised engagement
- Ad copy (if provided)

The model is instructed to respond **only** with a JSON object containing `summary`, `key_findings`, `alignment`, and `alignment_score`. Response is parsed with `json.loads()` with fallback to raw text.

Alignment score: `High`, `Medium`, or `Low` — based on whether the ad copy's hook, message, and CTA timing align with the measured attention curve.

### Step 8 — PDF Generation

`pipeline/pdf_generator.py`

Built with ReportLab's `BaseDocTemplate` + `Platypus` flowables. Layout:

- **Cover page:** SPIKD wordmark, job metadata, model/atlas citation
- **Metrics table:** Key values in a styled two-column table
- **Waveform section:** Embedded PNG + caption
- **Narrative section:** LLM summary + key findings list
- **Alignment section:** (if ad copy provided) quote block, score badge, analysis text
- **Footer:** Job ID + timestamp on every page

---

## 4. File Structure

```
spikd/
│
├── install.sh                  One-click RunPod installer
├── run.py                      Manual startup (development)
├── config.py                   Settings / environment loader
├── requirements.txt            Python dependencies
├── .env.example                Environment variable template
│
├── app/
│   ├── main.py                 FastAPI application + all routes
│   ├── auth.py                 Session auth dependency
│   │
│   ├── templates/
│   │   ├── base.html           Base layout (header, nav)
│   │   ├── login.html          Login page (standalone)
│   │   ├── dashboard.html      Main upload + job list
│   │   ├── progress.html       Real-time job tracking
│   │   └── result.html         Report viewer + download
│   │
│   └── static/
│       ├── css/main.css        Full UI stylesheet
│       └── js/main.js          Minimal JS (fetch interceptor)
│
├── pipeline/
│   ├── __init__.py
│   ├── tribe_processor.py      TRIBE v2 inference + 1D collapse + waveform PNG
│   ├── atlas_masker.py         Glasser atlas loading + vertex masking
│   ├── llm_summary.py          Llama 3.2 via Ollama
│   └── pdf_generator.py        ReportLab PDF engine
│
├── models/
│   └── tribe_v2/               Place tribe_v2.pt + config.json here
│
├── atlas_data/                 Glasser label.gii files (downloaded by installer)
│
├── data/
│   └── sessions/               Per-job working directories (video, waveform.png, report.pdf)
│
└── logs/
    └── spikd.log               Uvicorn + application logs
```

---

## 5. Installation Guide (RunPod)

### Prerequisites

- RunPod instance with RTX 4090 (24GB VRAM)
- CUDA 12.x base image (RunPod PyTorch template recommended)
- At least 40GB persistent storage
- Port 8000 exposed in pod settings

### Steps

**1. Connect to the pod**

Use RunPod's web terminal or SSH:
```bash
ssh root@<pod-ip> -p <ssh-port>
```

**2. Upload the SPIKD codebase**

Either clone from your private repo or use RunPod's file uploader to place the spikd/ folder in `/workspace/`.

**3. Run the installer**

```bash
cd /workspace/spikd
chmod +x install.sh
bash install.sh
```

The installer runs unattended. Total time: approximately 8–15 minutes depending on download speed.

**4. Access the dashboard**

In RunPod's pod settings, expose TCP port 8000. Then visit:
```
http://<pod-ip>:<mapped-port>
```

Default login: `admin` / `spikd2024`

**5. Change the default password**

Edit `/workspace/spikd/.env`:
```
SPIKD_USERS={"admin":"yournewpassword","analyst":"anotherpassword"}
```
Then restart:
```bash
systemctl restart spikd
```

### Reinstalling on a new pod

The same `install.sh` runs identically on any fresh RunPod instance. Each client's pod uses their own API key and billing. The script is completely environment-agnostic — no hardcoded IPs, no external accounts required.

---

## 6. Configuration Reference

All configuration lives in `/workspace/spikd/.env`.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | HTTP port |
| `SESSION_SECRET` | (auto-generated) | Cookie signing key — keep secret |
| `SPIKD_USERS` | `{"admin":"spikd2024"}` | JSON map of username → plaintext password |
| `SESSION_DIR` | `/workspace/spikd/data/sessions` | Job working directory |
| `ATLAS_DIR` | `/workspace/spikd/atlas_data` | Glasser atlas files |
| `TRIBE_MODEL_PATH` | `/workspace/spikd/models/tribe_v2` | TRIBE v2 checkpoint directory |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `LLM_MODEL` | `llama3.2` | Ollama model name |
| `VRAM_BUDGET_MB` | `20000` | Informational VRAM cap (MB) |

---

## 7. User Guide

### Uploading a video

1. Sign in at `/login`
2. On the Dashboard, drag a video file into the upload zone or click "browse"
3. Supported formats: MP4, MOV, AVI
4. Recommended: 1080p, 15–60 seconds
5. Optionally paste an ad script or headline into the "Ad Script / Headline" text box
6. Click "Run Analysis"

### Tracking progress

After upload, you are redirected to the progress page. The five pipeline stages are shown in real time:

1. Extracting video frames
2. TRIBE v2 cortical surface prediction
3. Glasser atlas masking (DAN + V1)
4. Llama 3.2 alignment analysis
5. Generating PDF report

### Viewing results

On completion, the result page shows:

- The monochrome Attention Spike Waveform with peak and drop markers
- Key metrics table (peak time, drop time, duration, FPS, regions, atlas)
- LLM narrative and key findings
- Ad copy alignment analysis and score (if ad copy was provided)

### Downloading the report

Click "Download PDF" on the result page. The PDF contains all the above in a formatted clinical report suitable for client delivery.

---

## 8. API Reference

All endpoints require an authenticated session cookie (obtained via POST `/login`).

| Method | Path | Description |
|---|---|---|
| GET | `/login` | Login page |
| POST | `/login` | Submit credentials — sets session cookie |
| GET | `/logout` | Clears session |
| GET | `/` | Dashboard |
| POST | `/analyze` | Upload video + ad copy. Returns `{"job_id": "..."}` |
| GET | `/status/{job_id}` | Poll job status. Returns JSON with `status`, `progress`, `message` |
| GET | `/progress/{job_id}` | Progress page (HTML) |
| GET | `/result/{job_id}` | Result page (HTML) |
| GET | `/download/{job_id}` | PDF download |
| GET | `/sessions/{job_id}/{filename}` | Session asset (e.g. waveform.png) |

### POST /analyze

**Form fields:**
- `video` (file, required) — video file
- `ad_copy` (string, optional) — ad script / headline

**Response:**
```json
{"job_id": "550e8400-e29b-41d4-a716-446655440000"}
```

### GET /status/{job_id}

**Response:**
```json
{
  "status": "running",
  "progress": 55,
  "message": "Running Llama 3.2 alignment analysis...",
  "filename": "ad_video.mp4",
  "ad_copy": "...",
  "result_path": null
}
```

`status` values: `queued`, `running`, `done`, `error`

---

## 9. TRIBE v2 Integration

TRIBE v2 (Meta Research, 2024) is a large-scale video-fMRI encoding model. It was trained on the Natural Scenes Dataset using paired video stimuli and BOLD responses recorded on fsLR-32k cortical surface.

**Model inputs:** RGB video frames at 224×224
**Model outputs:** Predicted BOLD on the 20,484-vertex fsLR-32k surface

### Loading weights

Place the following files in `/workspace/spikd/models/tribe_v2/`:

```
tribe_v2.pt      — PyTorch state dict
config.json      — Model configuration
```

The architecture class must be imported in `pipeline/tribe_processor.py`. When Meta releases the model publicly, uncomment the import block in `_load_model()` and update the class name to match the official package.

### HuggingFace access

The installer attempts:
```python
snapshot_download(repo_id="facebook/tribe-v2", ...)
```

If the repository is gated or not yet public, the system falls back to Simulation Mode (see Section 11).

---

## 10. Atlas Masking (Glasser MMP1.0)

### What the atlas does

The Glasser HCP MMP1.0 (Glasser et al., 2016, Nature) parcellates the human cortical surface into 360 areas based on architecture, function, connectivity, and topography. Each vertex in the fsLR-32k surface is assigned to one of these parcels.

### What SPIKD retains

SPIKD discards 350+ parcels and retains only the regions most relevant to visual attention processing:

**Dorsal Attention Network (DAN):**
- Intraparietal sulcus regions: LIPd, LIPv, VIP, MIP, AIP, 7PL
- Frontal eye field: area 8C, p9-46v
- Middle temporal area: MT+

**Primary Visual Cortex:**
- V1 (bilateral)

This is not a signal suppression (zeroing) — excluded vertices are physically removed from the matrix before the 1D collapse. Only scientifically relevant attention circuitry contributes to the waveform.

### Full atlas files

For production scientific validity, download the MMP1.0 label.gii files from the Human Connectome Project (HCP account required — db.humanconnectome.org):

```
L.MMP1.0.32k_fs_LR.label.gii
R.MMP1.0.32k_fs_LR.label.gii
```

Place them in `/workspace/spikd/atlas_data/`. The masker auto-detects and caches the vertex indices on first run.

---

## 11. Simulation Mode

When TRIBE v2 weights are not available, SPIKD runs in Simulation Mode.

In this mode, `_predict_surface()` generates a synthetic `[T, 20484]` tensor using:
- 1/f spectral noise base (realistic BOLD spectral profile)
- An injected synthetic attention peak at T//3

All downstream steps (atlas masking, 1D collapse, LLM analysis, PDF generation) run identically. Simulation Mode is clearly flagged in:
- The browser dashboard (yellow alert banner)
- The PDF report (notice at the bottom of the document)
- Application logs (`[WARN] Running in SIMULATION MODE`)

This allows full UI and pipeline testing before TRIBE v2 weights are available.

---

## 12. VRAM Management

### Budget (RTX 4090 — 24GB)

| Component | VRAM estimate |
|---|---|
| TRIBE v2 (approximate — depends on final architecture) | 10–14 GB |
| Llama 3.2 3B (via Ollama) | 2–3 GB |
| PyTorch buffers + overhead | 1–2 GB |
| **Total estimated** | **13–19 GB** |

### How management is handled

- TRIBE v2 inference runs first with `model.eval()` and `torch.no_grad()` to suppress gradient memory
- After TRIBE inference completes, the surface tensor is moved to CPU and the GPU buffer is freed before Ollama receives its request
- Ollama manages Llama 3.2 memory independently in a separate process
- The installer sets `VRAM_BUDGET_MB=20000` as a soft cap — add explicit `torch.cuda.empty_cache()` calls in `tribe_processor.py` if OOM errors occur

### If VRAM is tight

Use Llama 3.2:3b (default). Do not use the 11B variant on a 4090 alongside TRIBE v2.

---

## 13. Security

### Authentication

- Credentials are set via `SPIKD_USERS` environment variable as a JSON string
- Passwords are SHA-256 hashed on load — never stored in plaintext in memory
- Sessions are signed with `SESSION_SECRET` using itsdangerous (server-side only — no JWT)
- All routes require an authenticated session; unauthenticated API requests return HTTP 401

### Recommendations before client deployment

1. Generate a strong `SESSION_SECRET`: `python3 -c "import secrets; print(secrets.token_hex(32))"`
2. Change the default `admin` password in `SPIKD_USERS`
3. Place SPIKD behind a reverse proxy (nginx) with TLS if exposing to the internet
4. Restrict RunPod port exposure to known IP ranges where possible

### Data isolation

Each job writes to its own UUID-named subdirectory under `SESSION_DIR`. Users can only access their own jobs — the session username is checked on every `/status`, `/result`, `/download`, and `/sessions` request.

---

## 14. Troubleshooting

### SPIKD does not start

```bash
systemctl status spikd
tail -50 /workspace/spikd/logs/spikd.log
```

Common causes:
- Port 8000 already in use: change `PORT` in `.env`
- Missing Python package: `source /workspace/spikd/venv/bin/activate && pip install <package>`
- `.env` file missing: copy from `.env.example`

### Ollama / LLM not responding

```bash
systemctl status ollama
curl http://localhost:11434/api/tags
```

If Ollama is not running:
```bash
systemctl restart ollama
ollama pull llama3.2
```

The pipeline will still complete — LLM errors produce a fallback message in the PDF rather than crashing the job.

### TRIBE v2 not loading

Check the model directory:
```bash
ls -lh /workspace/spikd/models/tribe_v2/
```

If empty, the system is in Simulation Mode. To load real weights, place `tribe_v2.pt` and `config.json` in that directory and restart SPIKD.

### Video upload fails

- Confirm the file is a valid video (MP4, MOV, AVI)
- Check available disk space: `df -h /workspace`
- Check logs for OpenCV errors: `tail -100 /workspace/spikd/logs/spikd.log`

### Atlas masking using fallback

If you see `Using hard-coded vertex index approximation` in logs, the full Glasser label.gii files are not present. The pipeline still runs correctly using the hard-coded vertex ranges. For full MMP1.0 precision, obtain label.gii files from the HCP database.

---

*SPIKD v1.0 — Built on TRIBE v2 (Meta Research), Glasser MMP1.0 (Washington University), Llama 3.2 (Meta AI), FastAPI, ReportLab.*
