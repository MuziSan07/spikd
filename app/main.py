import os
import sys
import uuid
import asyncio
import shutil
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from app.auth import require_login, verify_credentials, RedirectException
from pipeline.tribe_processor import TribeProcessor
from pipeline.llm_summary import LLMSummary
from pipeline.pdf_generator import PDFGenerator

# ── App init ──────────────────────────────────────────────────
app = FastAPI(title="SPIKD", docs_url=None, redoc_url=None)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SESSION_SECRET,
    max_age=86400,
)

# ── Redirect unauthenticated users to /login ──────────────────
@app.exception_handler(RedirectException)
async def redirect_exception_handler(request: Request, exc: RedirectException):
    return RedirectResponse(url=exc.url, status_code=302)

BASE_DIR  = Path(__file__).parent
LOGO_PATH = Path("/workspace/spikd/app/static/agency_logo.png")  # persistent logo

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Shared pipeline instances (loaded once on startup)
tribe_processor: TribeProcessor = None
llm_summary: LLMSummary = None
pdf_generator: PDFGenerator = None

# In-memory job tracker
JOBS: dict = {}


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global tribe_processor, llm_summary, pdf_generator
    tribe_processor = TribeProcessor(model_path=settings.TRIBE_MODEL_PATH)
    llm_summary     = LLMSummary(ollama_host=settings.OLLAMA_HOST)
    pdf_generator   = PDFGenerator(output_dir=settings.SESSION_DIR)


# ── Auth routes ───────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    if verify_credentials(username, password):
        request.session["user"] = username
        return RedirectResponse(next, status_code=302)
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Invalid credentials.", "next": next}
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ── Dashboard ─────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(require_login)):
    recent_jobs = [
        {"job_id": jid, **meta}
        for jid, meta in JOBS.items()
        if meta.get("user") == user
    ][-10:]
    has_logo = LOGO_PATH.exists()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "recent_jobs": reversed(recent_jobs),
            "has_logo": has_logo,
        },
    )


# ── Logo upload ───────────────────────────────────────────────
@app.post("/upload-logo")
async def upload_logo(
    request: Request,
    logo: UploadFile = File(...),
    user: str = Depends(require_login),
):
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/svg+xml"}
    if logo.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Logo must be PNG, JPG, or SVG.")

    # Always save as PNG filename (works for png/jpg)
    LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGO_PATH, "wb") as f:
        content = await logo.read()
        f.write(content)

    return JSONResponse({"success": True, "message": "Logo uploaded successfully!"})


# ── Logo serve ────────────────────────────────────────────────
@app.get("/agency-logo")
async def serve_logo(user: str = Depends(require_login)):
    if LOGO_PATH.exists():
        return FileResponse(str(LOGO_PATH))
    raise HTTPException(status_code=404, detail="No logo uploaded.")


# ── Upload & process ──────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    request: Request,
    video: UploadFile = File(...),
    ad_copy: str = Form(""),
    user: str = Depends(require_login),
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    job_id  = str(uuid.uuid4())
    job_dir = Path(settings.SESSION_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / video.filename
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    JOBS[job_id] = {
        "user": user,
        "status": "queued",
        "progress": 0,
        "message": "Job queued.",
        "filename": video.filename,
        "ad_copy": ad_copy,
        "result_path": None,
        "llm_summary": "",
        "alignment": "",
        "alignment_score": "N/A",
        "peak_engagement_sec": "",
        "drop_timestamp_sec": "",
        "duration_sec": "",
        "fps": "",
        "n_frames": "",
        "key_findings": [],
    }

    asyncio.create_task(
        run_pipeline(job_id, str(video_path), ad_copy, user)
    )

    return JSONResponse({"job_id": job_id})


# ── Pipeline background task ──────────────────────────────────
async def run_pipeline(job_id: str, video_path: str, ad_copy: str, user: str):
    def update(progress: int, message: str, status: str = "running"):
        JOBS[job_id].update({"progress": progress, "message": message, "status": status})

    try:
        update(5, "Extracting video frames...")
        loop = asyncio.get_event_loop()

        # STEP 1 — TRIBE v2 brain prediction
        update(15, "Running TRIBE v2 cortical surface prediction...")
        brain_result = await loop.run_in_executor(
            None, tribe_processor.process, video_path
        )

        update(55, "Waveform extracted. Running Llama 3.2 alignment analysis...")

        # STEP 2 — LLM summary + alignment
        llm_result = await loop.run_in_executor(
            None, llm_summary.generate, brain_result, ad_copy
        )

        update(80, "Generating PDF report...")

        # STEP 3 — PDF (pass logo path)
        job_dir  = Path(settings.SESSION_DIR) / job_id
        logo_path = str(LOGO_PATH) if LOGO_PATH.exists() else None
        pdf_path = await loop.run_in_executor(
            None,
            pdf_generator.generate,
            job_id, job_dir, brain_result, llm_result, ad_copy, user, logo_path,
        )

        # ── Store all results back into JOBS ──
        JOBS[job_id]["result_path"]         = str(pdf_path)
        JOBS[job_id]["llm_summary"]         = llm_result.get("summary", "")
        JOBS[job_id]["alignment"]           = llm_result.get("alignment", "")
        JOBS[job_id]["alignment_score"]     = llm_result.get("alignment_score", "N/A")
        JOBS[job_id]["key_findings"]        = llm_result.get("key_findings", [])
        JOBS[job_id]["peak_engagement_sec"] = brain_result.get("peak_engagement_sec", "")
        JOBS[job_id]["drop_timestamp_sec"]  = brain_result.get("drop_timestamp_sec", "")
        JOBS[job_id]["duration_sec"]        = brain_result.get("duration_sec", "")
        JOBS[job_id]["fps"]                 = brain_result.get("fps", "")
        JOBS[job_id]["n_frames"]            = len(brain_result.get("waveform", []))

        update(100, "Complete.", "done")

    except Exception as exc:
        JOBS[job_id].update({"status": "error", "message": str(exc), "progress": 0})


# ── Polling endpoint ──────────────────────────────────────────
@app.get("/status/{job_id}")
async def job_status(job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(job)


# ── Result page ───────────────────────────────────────────────
@app.get("/result/{job_id}", response_class=HTMLResponse)
async def result_page(request: Request, job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        return RedirectResponse(f"/progress/{job_id}", status_code=302)

    waveform_img = f"/sessions/{job_id}/waveform.png"
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "user": user,
            "job": job,
            "job_id": job_id,
            "waveform_img": waveform_img,
        },
    )


# ── Progress page ─────────────────────────────────────────────
@app.get("/progress/{job_id}", response_class=HTMLResponse)
async def progress_page(request: Request, job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=404, detail="Job not found.")
    return templates.TemplateResponse(
        "progress.html", {"request": request, "user": user, "job_id": job_id}
    )


# ── PDF download ──────────────────────────────────────────────
@app.get("/download/{job_id}")
async def download_pdf(job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Report not available.")
    pdf_path = Path(job["result_path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file missing.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"SPIKD_Report_{job_id[:8]}.pdf",
    )


# ── Static session assets ─────────────────────────────────────
@app.get("/sessions/{job_id}/{filename}")
async def session_asset(job_id: str, filename: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=403)
    path = Path(settings.SESSION_DIR) / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(path))
