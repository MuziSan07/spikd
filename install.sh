#!/usr/bin/env bash
# ============================================================================
#  SPIKD v2 — One-Click Installer
#
#  Usage on any fresh RunPod instance:
#    curl -fsSL https://raw.githubusercontent.com/MuziSan07/spikd/master/install.sh | bash
#
#  OR:
#    wget -qO install.sh https://raw.githubusercontent.com/MuziSan07/spikd/master/install.sh
#    bash install.sh
#
#  The script will ask for your HuggingFace token.
#  That is the only input required. Everything else is automatic.
# ============================================================================

set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GRN}[SPIKD]${NC} $1"; }
warn() { echo -e "${YLW}[WARN] ${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

SPIKD_REPO="https://github.com/MuziSan07/spikd.git"
SPIKD_BRANCH="master"
SPIKD_DIR="/workspace/spikd"
VENV="$SPIKD_DIR/venv"
PORT="8000"

clear
echo ""
echo "  ███████╗██████╗ ██╗██╗  ██╗██████╗ "
echo "  ██╔════╝██╔══██╗██║██║ ██╔╝██╔══██╗"
echo "  ███████╗██████╔╝██║█████╔╝ ██║  ██║"
echo "  ╚════██║██╔═══╝ ██║██╔═██╗ ██║  ██║"
echo "  ███████║██║     ██║██║  ██╗██████╔╝"
echo "  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═════╝ "
echo ""
echo "  Cortical Attention Diagnostic Platform v2"
echo "  One-Click Installer"
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Ask for HuggingFace token ─────────────────────────────────────────────────
echo "  A HuggingFace token is required to download TRIBE v2 and Llama 3.2."
echo "  Generate one at: https://huggingface.co/settings/tokens"
echo "  Make sure you have accepted the terms for:"
echo "    - facebook/tribev2"
echo "    - meta-llama/Llama-3.2-3B"
echo ""
read -rsp "  Enter your HuggingFace token (input hidden): " HF_TOKEN
echo ""

if [[ -z "$HF_TOKEN" ]]; then
    die "No token provided. Exiting."
fi

export HF_TOKEN="$HF_TOKEN"
echo ""
log "Token received. Starting installation..."
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
log "Step 1/9 — Installing system packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    zstd ffmpeg git curl wget unzip \
    libgl1-mesa-glx libglib2.0-0 \
    fonts-liberation fonts-dejavu-core \
    python3-venv python3-dev build-essential \
    > /dev/null 2>&1
log "System packages installed."

# ── 2. Clone SPIKD repo ───────────────────────────────────────────────────────
log "Step 2/9 — Cloning SPIKD repository..."
if [[ -d "$SPIKD_DIR/.git" ]]; then
    log "Repository already cloned — pulling latest..."
    cd "$SPIKD_DIR" && git pull origin "$SPIKD_BRANCH" --quiet
else
    git clone -b "$SPIKD_BRANCH" "$SPIKD_REPO" "$SPIKD_DIR" --quiet
fi
mkdir -p "$SPIKD_DIR"/{data/sessions,atlas_data,models/tribe_v2,logs}
log "Repository ready at $SPIKD_DIR"

# ── 3. Ollama + Llama 3.2 ─────────────────────────────────────────────────────
log "Step 3/9 — Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh > /dev/null 2>&1
fi
pkill -f "ollama serve" 2>/dev/null || true
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 8
if ollama list 2>/dev/null | grep -q "llama3.2"; then
    log "Llama 3.2 already present."
else
    log "Pulling Llama 3.2 (~2GB)..."
    ollama pull llama3.2 || warn "Llama 3.2 pull failed. Run: ollama pull llama3.2"
fi

# ── 4. Python venv ────────────────────────────────────────────────────────────
log "Step 4/9 — Setting up Python environment..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel -q

log "Installing PyTorch (CUDA 12.1)..."
pip install -q torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

log "Installing application packages..."
pip install -q \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "itsdangerous==2.2.0" \
    "starlette==0.37.2" \
    "jinja2==3.1.4" \
    "opencv-python-headless" \
    "Pillow" \
    "numpy==1.26.4" \
    "scipy" \
    "nibabel" \
    "requests" \
    "reportlab" \
    "matplotlib" \
    "python-dotenv" \
    "huggingface_hub" \
    "uv"

# ── 5. HuggingFace login ──────────────────────────────────────────────────────
log "Step 5/9 — Authenticating with HuggingFace..."
python3 - <<PYEOF
from huggingface_hub import login
import os
try:
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    print("HuggingFace login successful.")
except Exception as e:
    print(f"HuggingFace login failed: {e}")
    exit(1)
PYEOF

# ── 6. TRIBE v2 ───────────────────────────────────────────────────────────────
log "Step 6/9 — Installing TRIBE v2..."
if [[ ! -d "/workspace/tribev2" ]]; then
    git clone https://github.com/facebookresearch/tribev2.git /workspace/tribev2 --quiet
fi
cd /workspace/tribev2
pip install -e . -q
cd "$SPIKD_DIR"

log "Downloading TRIBE v2 weights (~700MB)..."
python3 - <<PYEOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="facebook/tribev2",
    local_dir="/workspace/tribev2_weights",
    ignore_patterns=["*.msgpack"],
)
print("TRIBE v2 weights ready.")
PYEOF

log "Downloading Llama 3.2-3B config for TRIBE v2 text encoder..."
python3 - <<PYEOF
from huggingface_hub import snapshot_download
import os
try:
    snapshot_download(
        repo_id="meta-llama/Llama-3.2-3B",
        ignore_patterns=["*.bin", "*.safetensors", "original/*"],
    )
    print("Llama 3.2-3B config ready.")
except Exception as e:
    print(f"Warning: {e}")
    print("Visit https://huggingface.co/meta-llama/Llama-3.2-3B to request access.")
PYEOF

# ── 7. Atlas files ────────────────────────────────────────────────────────────
log "Step 7/9 — Downloading atlas files..."
ATLAS_DIR="$SPIKD_DIR/atlas_data"
HCP_GH="https://github.com/Washington-University/HCPpipelines/raw/master/global/templates/standard_mesh_atlases"
curl -fsSL -o "$ATLAS_DIR/L.atlasroi.32k_fs_LR.shape.gii" "$HCP_GH/L.atlasroi.32k_fs_LR.shape.gii" 2>/dev/null || true
curl -fsSL -o "$ATLAS_DIR/R.atlasroi.32k_fs_LR.shape.gii" "$HCP_GH/R.atlasroi.32k_fs_LR.shape.gii" 2>/dev/null || true
log "Atlas files ready."

# ── 8. Environment file ───────────────────────────────────────────────────────
log "Step 8/9 — Writing configuration..."
ENV_FILE="$SPIKD_DIR/.env"
SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
cat > "$ENV_FILE" <<ENVEOF
HOST=0.0.0.0
PORT=$PORT
SESSION_SECRET=$SECRET
SPIKD_USERS={"admin":"spikd2024"}
SESSION_DIR=$SPIKD_DIR/data/sessions
ATLAS_DIR=$SPIKD_DIR/atlas_data
TRIBE_MODEL_PATH=/workspace/tribev2_weights
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2
VRAM_BUDGET_MB=20000
HF_TOKEN=$HF_TOKEN
ENVEOF
log "Configuration written."

# ── 9. Start SPIKD ───────────────────────────────────────────────────────────
log "Step 9/9 — Starting SPIKD..."
source "$VENV/bin/activate"
export HF_TOKEN="$HF_TOKEN"
cd "$SPIKD_DIR"

nohup uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    > "$SPIKD_DIR/logs/spikd.log" 2>&1 &

sleep 8

HTTP=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/login" 2>/dev/null || echo "000")
HOST_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "═══════════════════════════════════════════════════════"
if [[ "$HTTP" == "200" ]]; then
    echo -e "  ${GRN}SPIKD v2 is live.${NC}"
else
    echo -e "  ${YLW}SPIKD starting up — check logs if it stays down.${NC}"
fi
echo ""
echo "  URL        :  http://$HOST_IP:$PORT"
echo "  RunPod     :  open port $PORT in pod HTTP settings"
echo ""
echo "  Username   :  admin"
echo "  Password   :  spikd2024"
echo ""
echo "  Logs       :  tail -f $SPIKD_DIR/logs/spikd.log"
echo "  Restart    :  cd $SPIKD_DIR && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"
echo "═══════════════════════════════════════════════════════"
echo ""
