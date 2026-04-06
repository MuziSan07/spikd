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

@app.exception_handler(RedirectException)
async def redirect_exception_handler(request: Request, exc: RedirectException):
    return RedirectResponse(url=exc.url, status_code=302)

BASE_DIR  = Path(__file__).parent
LOGO_PATH = Path("/workspace/spikd/app/static/agency_logo.png")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

tribe_processor: TribeProcessor = None
llm_summary: LLMSummary = None
pdf_generator: PDFGenerator = None

JOBS: dict = {}


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global tribe_processor, llm_summary, pdf_generator
    tribe_processor = TribeProcessor(model_path=settings.TRIBE_MODEL_PATH)
    llm_summary     = LLMSummary(ollama_host=settings.OLLAMA_HOST)
    pdf_generator   = PDFGenerator(output_dir=settings.SESSION_DIR)


# ── Auth ──────────────────────────────────────────────────────
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
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "recent_jobs": reversed(list(recent_jobs)),
            "has_logo": LOGO_PATH.exists(),
        },
    )


# ── Audit History ─────────────────────────────────────────────
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, user: str = Depends(require_login)):
    all_jobs = [
        {"job_id": jid, **meta}
        for jid, meta in JOBS.items()
        if meta.get("user") == user
    ]
    all_jobs.reverse()
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "user": user, "all_jobs": all_jobs},
    )


# ── Logo upload ───────────────────────────────────────────────
@app.post("/upload-logo")
async def upload_logo(
    request: Request,
    logo: UploadFile = File(...),
    user: str = Depends(require_login),
):
    allowed = {"image/png", "image/jpeg", "image/jpg"}
    if logo.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Logo must be PNG or JPG.")
    LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGO_PATH, "wb") as f:
        f.write(await logo.read())
    return JSONResponse({"success": True})


@app.get("/agency-logo")
async def serve_logo(user: str = Depends(require_login)):
    if LOGO_PATH.exists():
        return FileResponse(str(LOGO_PATH))
    raise HTTPException(status_code=404)


# ── Analyze ───────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    request: Request,
    video: UploadFile = File(...),
    ad_copy: str = Form(""),
    target_audience: str = Form(""),
    user: str = Depends(require_login),
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    job_id  = str(uuid.uuid4())
    job_dir = Path(settings.SESSION_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / video.filename
    with open(video_path, "wb") as f:
        f.write(await video.read())

    JOBS[job_id] = {
        "user": user,
        "status": "queued",
        "progress": 0,
        "message": "Job queued.",
        "filename": video.filename,
        "ad_copy": ad_copy,
        "target_audience": target_audience,
        "result_path": None,
        "llm_summary": "",
        "alignment": "",
        "alignment_score": "N/A",
        "key_findings": [],
        "recommendations": [],
        "peak_engagement_sec": "",
        "drop_timestamp_sec": "",
        "duration_sec": "",
        "fps": "",
        "n_frames": "",
        "peak_frame_path": None,
    }

    asyncio.create_task(run_pipeline(job_id, str(video_path), ad_copy, target_audience, user))
    return JSONResponse({"job_id": job_id})


# ── Compare ───────────────────────────────────────────────────
@app.post("/compare")
async def compare(
    request: Request,
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    user: str = Depends(require_login),
):
    for v in [video_a, video_b]:
        if not v.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Both files must be videos.")

    job_a = str(uuid.uuid4())
    job_b = str(uuid.uuid4())

    for job_id, video in [(job_a, video_a), (job_b, video_b)]:
        job_dir = Path(settings.SESSION_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        video_path = job_dir / video.filename
        with open(video_path, "wb") as f:
            f.write(await video.read())
        JOBS[job_id] = {
            "user": user, "status": "queued", "progress": 0,
            "message": "Job queued.", "filename": video.filename,
            "ad_copy": "", "target_audience": "", "result_path": None,
            "llm_summary": "", "alignment": "", "alignment_score": "N/A",
            "key_findings": [], "recommendations": [],
            "peak_engagement_sec": "", "drop_timestamp_sec": "",
            "duration_sec": "", "fps": "", "n_frames": "",
            "peak_frame_path": None,
        }
        asyncio.create_task(run_pipeline(job_id, str(video_path), "", "", user))

    return JSONResponse({"job_a": job_a, "job_b": job_b})


# ── Pipeline ──────────────────────────────────────────────────
async def run_pipeline(job_id: str, video_path: str, ad_copy: str, target_audience: str, user: str):
    def update(progress: int, message: str, status: str = "running"):
        JOBS[job_id].update({"progress": progress, "message": message, "status": status})

    try:
        update(5, "Extracting video frames...")
        loop = asyncio.get_event_loop()

        update(15, "Running cortical surface prediction...")
        brain_result = await loop.run_in_executor(None, tribe_processor.process, video_path)

        # ── Save peak frame as JPG ──
        peak_frame_path = None
        try:
            import cv2
            peak_sec = brain_result.get("peak_engagement_sec", 0)
            fps      = brain_result.get("fps", 24)
            peak_frame_idx = int(peak_sec * fps)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                job_dir = Path(settings.SESSION_DIR) / job_id
                peak_frame_path = str(job_dir / "peak_frame.jpg")
                cv2.imwrite(peak_frame_path, frame)
        except Exception as e:
            pass

        update(55, "Running AI alignment analysis...")
        llm_result = await loop.run_in_executor(
            None, llm_summary.generate, brain_result, ad_copy, target_audience
        )

        update(80, "Generating PDF report...")
        job_dir   = Path(settings.SESSION_DIR) / job_id
        logo_path = str(LOGO_PATH) if LOGO_PATH.exists() else None
        pdf_path  = await loop.run_in_executor(
            None, pdf_generator.generate,
            job_id, job_dir, brain_result, llm_result, ad_copy, target_audience, user, logo_path, peak_frame_path,
        )

        JOBS[job_id].update({
            "result_path":         str(pdf_path),
            "llm_summary":         llm_result.get("summary", ""),
            "alignment":           llm_result.get("alignment", ""),
            "alignment_score":     llm_result.get("alignment_score", "N/A"),
            "key_findings":        llm_result.get("key_findings", []),
            "recommendations":     llm_result.get("recommendations", []),
            "peak_engagement_sec": brain_result.get("peak_engagement_sec", ""),
            "drop_timestamp_sec":  brain_result.get("drop_timestamp_sec", ""),
            "duration_sec":        brain_result.get("duration_sec", ""),
            "fps":                 brain_result.get("fps", ""),
            "n_frames":            len(brain_result.get("waveform", [])),
            "peak_frame_path":     peak_frame_path,
        })
        update(100, "Complete.", "done")

    except Exception as exc:
        JOBS[job_id].update({"status": "error", "message": str(exc), "progress": 0})


# ── Status / Result / Progress ────────────────────────────────
@app.get("/status/{job_id}")
async def job_status(job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=404)
    return JSONResponse(job)


@app.get("/result/{job_id}", response_class=HTMLResponse)
async def result_page(request: Request, job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=404)
    if job["status"] != "done":
        return RedirectResponse(f"/progress/{job_id}", status_code=302)
    has_peak_frame = bool(job.get("peak_frame_path")) and Path(job["peak_frame_path"]).exists()
    return templates.TemplateResponse("result.html", {
        "request": request, "user": user, "job": job,
        "job_id": job_id,
        "waveform_img": f"/sessions/{job_id}/waveform.png",
        "peak_frame_img": f"/sessions/{job_id}/peak_frame.jpg" if has_peak_frame else None,
    })


@app.get("/compare/{job_a}/{job_b}", response_class=HTMLResponse)
async def compare_page(request: Request, job_a: str, job_b: str, user: str = Depends(require_login)):
    ja = JOBS.get(job_a)
    jb = JOBS.get(job_b)
    if not ja or not jb or ja["user"] != user or jb["user"] != user:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse("compare.html", {
        "request": request, "user": user,
        "job_a": ja, "job_b": jb,
        "job_a_id": job_a, "job_b_id": job_b,
        "waveform_a": f"/sessions/{job_a}/waveform.png",
        "waveform_b": f"/sessions/{job_b}/waveform.png",
    })


@app.get("/progress/{job_id}", response_class=HTMLResponse)
async def progress_page(request: Request, job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse("progress.html", {"request": request, "user": user, "job_id": job_id})


@app.get("/download/{job_id}")
async def download_pdf(job_id: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user or job["status"] != "done":
        raise HTTPException(status_code=404)
    pdf_path = Path(job["result_path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(pdf_path), media_type="application/pdf",
                        filename=f"SPIKD_Report_{job_id[:8]}.pdf")


@app.get("/sessions/{job_id}/{filename}")
async def session_asset(job_id: str, filename: str, user: str = Depends(require_login)):
    job = JOBS.get(job_id)
    if not job or job["user"] != user:
        raise HTTPException(status_code=403)
    path = Path(settings.SESSION_DIR) / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(path))
