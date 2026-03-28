"""
pipeline/pdf_generator.py

Clinical Instrument PDF output.

Design spec:
  - Pitch black background throughout (#000000)
  - Monospace font (Courier) for all text
  - Page 1: TOP HALF = waveform graph (electric white line, no fill)
            BOTTOM HALF = Llama 3.2 analysis text
  - Page 2: Alignment section (only if ad copy was provided)
  - Footer: job ID + timestamp on every page
  - No gradients, no colour, no decorative elements
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import ParagraphStyle

logger = logging.getLogger("spikd.pdf")

# ── Palette ───────────────────────────────────────────────────────────────────
BLACK     = colors.HexColor("#000000")
MID_GREY  = colors.HexColor("#333333")
DIM_WHITE = colors.HexColor("#777777")
WHITE     = colors.HexColor("#ffffff")

# ── Fonts — monospace only ────────────────────────────────────────────────────
MONO      = "Courier"
MONO_BOLD = "Courier-Bold"

W, H = A4


class PDFGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    # ── Public entry point ─────────────────────────────────────────────────────
    def generate(
        self,
        job_id: str,
        job_dir: Path,
        brain_result: dict,
        llm_result: dict,
        ad_copy: str,
        user: str,
    ) -> Path:
        """
        Builds the PDF.
        Page 1: header strip / waveform (top) / metrics line / LLM analysis (bottom)
        Page 2: alignment analysis (only if ad copy provided)
        """
        out_path = job_dir / "report.pdf"
        styles   = self._styles()

        doc = BaseDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
        )
        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="main",
        )
        doc.addPageTemplates([
            PageTemplate(
                id="main",
                frames=[frame],
                onPage=self._draw_page(job_id),
            )
        ])

        story = []

        # ── Header strip ──────────────────────────────────────────────────────────
        ts = datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")
        story.append(Paragraph(
            f"SPIKD  //  CORTICAL ATTENTION REPORT  //  "
            f"JOB {job_id[:16].upper()}  //  {ts}",
            styles["header"],
        ))
        story.append(Spacer(1, 3 * mm))
        story.append(HRFlowable(width="100%", thickness=0.4, color=MID_GREY))
        story.append(Spacer(1, 4 * mm))

        # ── TOP HALF — waveform image ─────────────────────────────────────────────
        waveform_png = Path(brain_result.get("waveform_png_path", ""))
        if waveform_png.exists():
            img_w = doc.width
            img_h = img_w * (4 / 12)   # matches figsize(12, 4) render ratio
            story.append(Image(str(waveform_png), width=img_w, height=img_h))
        else:
            story.append(Paragraph("[WAVEFORM UNAVAILABLE]", styles["label"]))

        story.append(Spacer(1, 3 * mm))

        # ── Compact metrics line ──────────────────────────────────────────────────
        wf  = brain_result.get("waveform", [])
        avg = round(float(np.mean(wf)), 3) if wf else 0.0
        story.append(Paragraph(
            f"PEAK {brain_result.get('peak_engagement_sec', 0):.2f}s  |  "
            f"DROP {brain_result.get('drop_timestamp_sec', 0):.2f}s  |  "
            f"AVG {avg:.3f}  |  "
            f"DUR {brain_result.get('duration_sec', 0):.1f}s  |  "
            f"FPS {brain_result.get('fps', 0):.1f}  |  "
            f"REGIONS: DAN + V1  |  ATLAS: GLASSER MMP1.0",
            styles["metrics"],
        ))
        story.append(Spacer(1, 5 * mm))
        story.append(HRFlowable(width="100%", thickness=0.4, color=MID_GREY))
        story.append(Spacer(1, 5 * mm))

        # ── BOTTOM HALF — LLM analysis ────────────────────────────────────────────
        story.append(Paragraph("NEURAL ENGAGEMENT ANALYSIS", styles["label"]))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            llm_result.get("summary", "Analysis unavailable."),
            styles["body"],
        ))

        findings = llm_result.get("key_findings", [])
        if findings:
            story.append(Spacer(1, 5 * mm))
            story.append(Paragraph("KEY FINDINGS", styles["label"]))
            story.append(Spacer(1, 2 * mm))
            for f in findings:
                story.append(Paragraph(f"--  {f}", styles["bullet"]))

        # ── Simulation warning ────────────────────────────────────────────────────
        if brain_result.get("simulation_mode"):
            story.append(Spacer(1, 6 * mm))
            story.append(HRFlowable(width="100%", thickness=0.3, color=MID_GREY))
            story.append(Spacer(1, 3 * mm))
            story.append(Paragraph(
                "[ SIMULATION MODE — TRIBE v2 WEIGHTS NOT LOADED — "
                "SYNTHETIC DATA ONLY — NOT FOR CLINICAL USE ]",
                styles["notice"],
            ))

        # ── Page 2 — alignment ────────────────────────────────────────────────────
        if ad_copy.strip():
            story.append(PageBreak())
            story += self._alignment_page(styles, llm_result, ad_copy, job_id, ts)

        doc.build(story)
        logger.info("PDF report written: %s", out_path)
        return out_path

    # ── Alignment page ─────────────────────────────────────────────────────────
    def _alignment_page(self, styles, llm_result, ad_copy, job_id, ts):
        score = llm_result.get("alignment_score", "N/A").upper()
        items = []

        items.append(Paragraph(
            f"SPIKD  //  AD COPY ALIGNMENT  //  JOB {job_id[:16].upper()}  //  {ts}",
            styles["header"],
        ))
        items.append(Spacer(1, 3 * mm))
        items.append(HRFlowable(width="100%", thickness=0.4, color=MID_GREY))
        items.append(Spacer(1, 6 * mm))

        items.append(Paragraph("AD COPY SUBMITTED", styles["label"]))
        items.append(Spacer(1, 2 * mm))
        items.append(Paragraph(ad_copy, styles["quote"]))
        items.append(Spacer(1, 6 * mm))

        items.append(HRFlowable(width="100%", thickness=0.3, color=MID_GREY))
        items.append(Spacer(1, 5 * mm))

        items.append(Paragraph(f"ALIGNMENT SCORE:  {score}", styles["score"]))
        items.append(Spacer(1, 5 * mm))

        items.append(Paragraph("ALIGNMENT ANALYSIS", styles["label"]))
        items.append(Spacer(1, 3 * mm))
        items.append(Paragraph(
            llm_result.get("alignment", "No alignment analysis available."),
            styles["body"],
        ))

        return items

    # ── Page background + footer ───────────────────────────────────────────────
    def _draw_page(self, job_id: str):
        def on_page(canvas, doc):
            canvas.saveState()
            # Pitch black page background
            canvas.setFillColor(BLACK)
            canvas.rect(0, 0, W, H, fill=1, stroke=0)
            # Footer
            canvas.setFont(MONO, 6.5)
            canvas.setFillColor(DIM_WHITE)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            canvas.drawString(18 * mm, 9 * mm, f"SPIKD  |  {job_id[:24]}  |  {ts}")
            canvas.drawRightString(W - 18 * mm, 9 * mm, f"PAGE {doc.page}")
            canvas.restoreState()
        return on_page

    # ── Styles — monospace, white on black ─────────────────────────────────────
    def _styles(self) -> dict:
        return {
            "header": ParagraphStyle(
                "header",
                fontName=MONO, fontSize=7,
                textColor=DIM_WHITE,
                leading=11, spaceAfter=0,
                alignment=TA_LEFT,
                backColor=BLACK,
            ),
            "label": ParagraphStyle(
                "label",
                fontName=MONO_BOLD, fontSize=8,
                textColor=WHITE,
                leading=12, spaceAfter=2,
                alignment=TA_LEFT,
                letterSpacing=1.5,
                backColor=BLACK,
            ),
            "metrics": ParagraphStyle(
                "metrics",
                fontName=MONO, fontSize=7,
                textColor=DIM_WHITE,
                leading=11, spaceAfter=0,
                alignment=TA_LEFT,
                backColor=BLACK,
            ),
            "body": ParagraphStyle(
                "body",
                fontName=MONO, fontSize=8.5,
                textColor=WHITE,
                leading=14, spaceAfter=3,
                alignment=TA_JUSTIFY,
                backColor=BLACK,
            ),
            "bullet": ParagraphStyle(
                "bullet",
                fontName=MONO, fontSize=8.5,
                textColor=WHITE,
                leading=13, spaceAfter=3,
                leftIndent=12,
                backColor=BLACK,
            ),
            "quote": ParagraphStyle(
                "quote",
                fontName=MONO, fontSize=8,
                textColor=DIM_WHITE,
                leading=13, spaceAfter=3,
                leftIndent=10,
                alignment=TA_LEFT,
                backColor=BLACK,
            ),
            "score": ParagraphStyle(
                "score",
                fontName=MONO_BOLD, fontSize=13,
                textColor=WHITE,
                leading=18, spaceAfter=4,
                alignment=TA_LEFT,
                backColor=BLACK,
            ),
            "notice": ParagraphStyle(
                "notice",
                fontName=MONO, fontSize=7,
                textColor=DIM_WHITE,
                leading=11, spaceAfter=0,
                alignment=TA_CENTER,
                backColor=BLACK,
            ),
        }
