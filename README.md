# SPIKD

SPIKD is a neural-attention analytics pipeline for video advertising. It predicts
where human cortical attention is likely to concentrate across a video ad,
flags low-attention segments ("attention flatlines"), and produces a PDF report
to help reduce wasted ad spend on weak hooks.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C.svg)](https://pytorch.org/)

---

## How it works

SPIKD combines two engines:

1. **Cortical predictor** — the TRIBE v2 (Trimodal Brain Encoder) model maps
   visual saliency against a baseline derived from fMRI neuro-imaging data, using
   the Glasser cortical atlas for region masking.
2. **Semantic auditor** — a Llama 3.2 model (served via Ollama) cross-references
   attention flatlines with the ad's script to explain likely cognitive overload.

The result is rendered into a frame-by-frame attention report (PDF).

---

## Features

- Web UI (FastAPI + Jinja2) with session-based login
- Upload a video and run attention prediction frame by frame
- TRIBE v2 cortical attention inference with Glasser-atlas masking
- Llama 3.2 semantic summary of weak segments
- PDF telemetry report with waveform and per-frame peaks
- Compare mode for evaluating two creatives against a target audience
- Job history and progress tracking

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Web framework | FastAPI, Uvicorn, Starlette sessions |
| Templating | Jinja2 |
| ML / inference | PyTorch, TRIBE v2, transformers |
| LLM | Llama 3.2 via Ollama |
| Video / image | OpenCV, Pillow |
| Neuroimaging | nibabel (Glasser atlas) |
| Reporting | ReportLab, matplotlib |

---

## Project structure

```
spikd/
├── run.py                      # Development entry point (uvicorn)
├── config.py                   # Settings, loaded from environment / .env
├── requirements.txt
├── install.sh                  # Server provisioning script (RunPod / Linux)
├── .env.example                # Environment template
├── DOCUMENTATION.md            # Full deployment and operations guide
├── app/
│   ├── main.py                 # FastAPI app and routes
│   ├── auth.py                 # Session authentication helpers
│   ├── static/                 # CSS and JavaScript
│   └── templates/              # Jinja2 HTML templates
└── pipeline/
    ├── tribe_processor.py      # TRIBE v2 cortical attention inference
    ├── atlas_masker.py         # Glasser atlas vertex masking
    ├── llm_summary.py          # Llama 3.2 (Ollama) semantic summary
    └── pdf_generator.py        # PDF report generation
```

---

## Prerequisites

- Python 3.10 or newer
- A CUDA-capable GPU is strongly recommended for TRIBE v2 inference
- [Ollama](https://ollama.com/) running the `llama3.2` model for the semantic auditor
- The TRIBE v2 model weights and Glasser atlas files (see `DOCUMENTATION.md`)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MuziSan07/spikd.git
cd spikd

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install PyTorch for your platform (see https://pytorch.org/get-started/)
#    then the remaining dependencies
pip install -r requirements.txt

# 4. Configure the environment
cp .env.example .env
# Edit .env: set a strong SESSION_SECRET and your credentials

# 5. Run
python run.py
```

The app listens on `http://<HOST>:<PORT>` (default `0.0.0.0:8000`).

For automated server setup (model download, atlas, systemd service), see
`install.sh` and `DOCUMENTATION.md`.

---

## Configuration

All settings are read from environment variables (via `.env`).

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Bind address | `0.0.0.0` |
| `PORT` | Port | `8000` |
| `SESSION_SECRET` | Cookie signing key. **Required for production** — a temporary random key is generated (with a warning) if unset. | (generated) |
| `SPIKD_USERS` | JSON map of `{"user": "password"}`; stored hashed | `{"admin": "spikd2024"}` |
| `ADMIN_USER` / `ADMIN_PASS` | Single default user, if `SPIKD_USERS` is unset | `admin` / `spikd2024` |
| `SESSION_DIR` | Job working directory | `./data/sessions` |
| `ATLAS_DIR` | Glasser atlas files | `./atlas_data` |
| `TRIBE_MODEL_PATH` | TRIBE v2 checkpoint directory | `./models/tribe_v2` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `LLM_MODEL` | Ollama model name | `llama3.2` |
| `VRAM_BUDGET_MB` | Informational VRAM budget | `20000` |

---

## Usage

1. Sign in with your configured credentials.
2. Upload a video creative.
3. Run the pipeline and watch progress.
4. Review the attention report and download the PDF.
5. Optionally use compare mode to evaluate two creatives.

---

## Security notes

- `SESSION_SECRET` has no insecure default: if it is unset or left at the
  placeholder value, the app generates a random per-process key and warns. Set a
  stable secret in `.env` for production (`python3 -c "import secrets; print(secrets.token_hex(32))"`).
- Change the default `admin / spikd2024` credentials before exposing the service;
  the app warns while the default is in use.
- `.env`, session data, logs, and model weights are git-ignored.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
