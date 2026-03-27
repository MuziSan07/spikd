#!/usr/bin/env bash
# ============================================================================
#  SPIKD — One-Click Installer for RunPod (RTX 4090 / CUDA 12.x)
#
#  Usage on a fresh RunPod instance:
#    chmod +x install.sh && bash install.sh
#
#  What this does:
#    1. System dependencies (ffmpeg, fonts, git)
#    2. Ollama + Llama 3.2
#    3. Python venv + all packages (PyTorch CUDA 12.1, FastAPI, ReportLab...)
#    4. Glasser HCP MMP1.0 atlas files
#    5. TRIBE v2 weight download attempt (HuggingFace)
#    6. .env configuration
#    7. systemd services (spikd + ollama)
#    8. Starts dashboard on port 8000
#
#  All GPU compute and storage costs run on the client's RunPod account.
# ============================================================================

set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GRN}[SPIKD]${NC} $1"; }
warn() { echo -e "${YLW}[WARN] ${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Configurable vars ─────────────────────────────────────────────────────────
SPIKD_DIR="${SPIKD_DIR:-/workspace/spikd}"
VENV="$SPIKD_DIR/venv"
PORT="${PORT:-8000}"
CUDA_URL="https://download.pytorch.org/whl/cu121"
HF_TRIBE_REPO="${HF_TRIBE_REPO:-facebook/tribe-v2}"

echo ""
echo "  ███████╗██████╗ ██╗██╗  ██╗██████╗ "
echo "  ██╔════╝██╔══██╗██║██║ ██╔╝██╔══██╗"
echo "  ███████╗██████╔╝██║█████╔╝ ██║  ██║"
echo "  ╚════██║██╔═══╝ ██║██╔═██╗ ██║  ██║"
echo "  ███████║██║     ██║██║  ██╗██████╔╝"
echo "  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═════╝ "
echo ""
log "Cortical Attention Diagnostic Platform — Installer v1.0"
echo ""

# ── 0. Pre-flight checks ──────────────────────────────────────────────────────
log "Running pre-flight checks..."
[[ $EUID -eq 0 ]] || die "Run as root: sudo bash install.sh"
command -v python3 &>/dev/null || die "python3 not found."
command -v pip3    &>/dev/null || die "pip3 not found."
command -v git     &>/dev/null || die "git not found."
command -v curl    &>/dev/null || die "curl not found."

GPU_OK=$(nvidia-smi &>/dev/null && echo "yes" || echo "no")
[[ "$GPU_OK" == "yes" ]] && log "GPU detected." || warn "nvidia-smi not responding — GPU may not be ready yet."

FREE_KB=$(df /workspace 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
FREE_GB=$(( FREE_KB / 1048576 ))
[[ $FREE_GB -lt 30 ]] && warn "Only ${FREE_GB}GB free. Recommend 40GB+ for model weights."

# ── 1. System packages ────────────────────────────────────────────────────────
log "Installing system packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    curl wget git unzip \
    fonts-liberation fonts-dejavu-core \
    python3-venv python3-dev build-essential \
    > /dev/null 2>&1
log "System packages installed."

# ── 2. Ollama + Llama 3.2 ─────────────────────────────────────────────────────
log "Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh > /dev/null 2>&1
fi

pkill -f "ollama serve" 2>/dev/null || true
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 6

if ollama list 2>/dev/null | grep -q "llama3.2"; then
    log "Llama 3.2 already present."
else
    log "Pulling Llama 3.2 (~2GB)..."
    ollama pull llama3.2 || warn "Llama 3.2 pull failed. Run: ollama pull llama3.2"
fi

# ── 3. SPIKD directory ────────────────────────────────────────────────────────
log "Setting up $SPIKD_DIR..."
mkdir -p "$SPIKD_DIR"/{data/sessions,atlas_data,models/tribe_v2,logs}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" != "$SPIKD_DIR" ]]; then
    log "Copying application files..."
    cp -r "$SCRIPT_DIR"/. "$SPIKD_DIR"/
fi

# ── 4. Python virtual environment ─────────────────────────────────────────────
log "Creating Python virtual environment..."
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip setuptools wheel -q

log "Installing PyTorch with CUDA 12.1..."
pip install -q torch torchvision torchaudio --index-url "$CUDA_URL"

log "Installing application dependencies..."
pip install -q \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "itsdangerous==2.2.0" \
    "starlette==0.37.2" \
    "jinja2==3.1.4" \
    "opencv-python-headless==4.9.0.80" \
    "Pillow==10.3.0" \
    "numpy==1.26.4" \
    "scipy==1.13.0" \
    "nibabel==5.2.0" \
    "requests==2.32.0" \
    "reportlab==4.2.0" \
    "matplotlib==3.9.0" \
    "python-dotenv==1.0.1" \
    "huggingface_hub==0.23.0" \
    "transformers==4.41.0" \
    "accelerate==0.30.0" \
    "einops==0.7.0" \
    "timm==1.0.3"

deactivate
log "Python environment ready."

# ── 5. Atlas files ────────────────────────────────────────────────────────────
log "Downloading Glasser HCP MMP1.0 atlas files..."
ATLAS_DIR="$SPIKD_DIR/atlas_data"

dl_atlas() {
    local url="$1" dest="$2"
    [[ -f "$dest" ]] && { log "  Present: $(basename "$dest")"; return; }
    curl -fsSL -o "$dest" "$url" 2>/dev/null \
        && log "  Downloaded: $(basename "$dest")" \
        || warn "  Could not fetch $(basename "$dest") — hard-coded fallback will be used."
}

HCP_GH="https://github.com/Washington-University/HCPpipelines/raw/master/global/templates/standard_mesh_atlases"
dl_atlas "$HCP_GH/L.atlasroi.32k_fs_LR.shape.gii" "$ATLAS_DIR/L.atlasroi.32k_fs_LR.shape.gii"
dl_atlas "$HCP_GH/R.atlasroi.32k_fs_LR.shape.gii" "$ATLAS_DIR/R.atlasroi.32k_fs_LR.shape.gii"

warn "Full MMP1.0 label.gii files require HCP account (db.humanconnectome.org)."
warn "Place L/R.MMP1.0.32k_fs_LR.label.gii in $ATLAS_DIR for production atlas masking."

# ── 6. TRIBE v2 weights ───────────────────────────────────────────────────────
log "Attempting TRIBE v2 download from HuggingFace ($HF_TRIBE_REPO)..."
source "$VENV/bin/activate"
python3 - <<PYEOF 2>&1 | tail -8 || true
from huggingface_hub import snapshot_download, list_repo_files
import sys
try:
    list(list_repo_files("$HF_TRIBE_REPO"))  # will raise if repo not accessible
    snapshot_download(
        repo_id="$HF_TRIBE_REPO",
        local_dir="$SPIKD_DIR/models/tribe_v2",
        ignore_patterns=["*.msgpack", "flax_model*"],
    )
    print("TRIBE v2 weights downloaded.")
except Exception as exc:
    print(f"TRIBE v2 not accessible: {exc}")
    print("System will start in SIMULATION MODE.")
    print("Place tribe_v2.pt + config.json in: $SPIKD_DIR/models/tribe_v2")
    sys.exit(0)
PYEOF
deactivate

# ── 7. Environment file ───────────────────────────────────────────────────────
log "Writing .env..."
ENV_FILE="$SPIKD_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    cat > "$ENV_FILE" <<ENVEOF
HOST=0.0.0.0
PORT=$PORT
SESSION_SECRET=$SECRET
SPIKD_USERS={"admin":"spikd2024"}
SESSION_DIR=$SPIKD_DIR/data/sessions
ATLAS_DIR=$SPIKD_DIR/atlas_data
TRIBE_MODEL_PATH=$SPIKD_DIR/models/tribe_v2
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2
VRAM_BUDGET_MB=20000
ENVEOF
    log ".env created. Default credentials: admin / spikd2024"
    warn "Change SPIKD_USERS in $ENV_FILE before exposing to the internet."
else
    log ".env already exists — skipping."
fi

# ── 8. systemd services ───────────────────────────────────────────────────────
log "Registering systemd services..."

cat > /etc/systemd/system/ollama.service <<OLEOF
[Unit]
Description=Ollama LLM service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
OLEOF

cat > /etc/systemd/system/spikd.service <<SVCEOF
[Unit]
Description=SPIKD — Cortical Attention Diagnostic Platform
After=network.target ollama.service

[Service]
Type=simple
User=root
WorkingDirectory=$SPIKD_DIR
EnvironmentFile=$SPIKD_DIR/.env
ExecStart=$VENV/bin/uvicorn app.main:app \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --workers 1 \\
    --log-level info
Restart=on-failure
RestartSec=10
StandardOutput=append:$SPIKD_DIR/logs/spikd.log
StandardError=append:$SPIKD_DIR/logs/spikd.log

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable ollama.service
systemctl restart ollama.service || true
sleep 3
systemctl enable spikd.service
systemctl restart spikd.service || true

# ── 9. Verify ─────────────────────────────────────────────────────────────────
log "Waiting for SPIKD to start..."
sleep 7

HTTP=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/login" 2>/dev/null || echo "000")
HOST_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "═══════════════════════════════════════════════════════════"
if [[ "$HTTP" == "200" ]]; then
    echo -e "  ${GRN}SPIKD is running — HTTP $HTTP${NC}"
else
    echo -e "  ${YLW}SPIKD starting (HTTP $HTTP) — check logs if it stays at this.${NC}"
fi
echo ""
echo "  Dashboard  :  http://$HOST_IP:$PORT"
echo "  RunPod     :  expose TCP port $PORT in pod settings"
echo ""
echo "  Login      :  admin / spikd2024"
echo "  Change pw  :  edit SPIKD_USERS in $ENV_FILE then: systemctl restart spikd"
echo ""
echo "  Logs       :  tail -f $SPIKD_DIR/logs/spikd.log"
echo "  Restart    :  systemctl restart spikd"
echo "  Stop       :  systemctl stop spikd"
echo ""
echo "  Model dir  :  $SPIKD_DIR/models/tribe_v2/"
echo "  Atlas dir  :  $SPIKD_DIR/atlas_data/"
echo "═══════════════════════════════════════════════════════════"
echo ""
