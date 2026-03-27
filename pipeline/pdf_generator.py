"""
pipeline/pdf_generator.py

Generates a professional clinical-style PDF report using ReportLab.
Layout:
  - Cover page with SPIKD branding and metadata
  - Brain Engagement Summary (waveform PNG embedded)
  - Key Metrics table
  - LLM Narrative section
  - Ad Copy Alignment section (if applicable)
  - Footer with timestamp and job ID
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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
    Table,
    TableStyle,
)

logger = logging.getLogger("spikd.pdf")

# ── Colour palette (monochrome / dark report) ──────────────────────────────────
BLACK      = colors.HexColor("#0a0a0a")
DARK_GREY  = colors.HexColor("#1a1a1a")
MID_GREY   = colors.HexColor("#555555")
LIGHT_GREY = colors.HexColor("#cccccc")
OFF_WHITE  = colors.HexColor("#f2f2f2")
WHITE      = colors.white

W, H = A4


class PDFGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def generate(
        self,
        job_id: str,
        job_dir: Path,
        brain_result: dict,
        llm_result: dict,
        ad_copy: str,
        user: str,
    ) -> Path:
        out_path = job_dir / "report.pdf"
        styles   = self._build_styles()

        doc = BaseDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=25 * mm,
            bottomMargin=20 * mm,
        )

        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="main",
        )
        page_template = PageTemplate(
            id="main",
            frames=[frame],
            onPage=self._page_footer(job_id),
        )
        doc.addPageTemplates([page_template])

        story = []

        # ── Cover ───────────────────────────────────────────────────────────────
        story += self._build_cover(styles, job_id, user, brain_result)
        story.append(PageBreak())

        # ── Metrics ─────────────────────────────────────────────────────────────
        story += self._build_metrics(styles, brain_result)
        story.append(Spacer(1, 6 * mm))

        # ── Waveform ────────────────────────────────────────────────────────────
        waveform_png = Path(brain_result.get("waveform_png_path", ""))
        if waveform_png.exists():
            story += self._build_waveform_section(styles, waveform_png, brain_result)
        story.append(Spacer(1, 6 * mm))

        # ── LLM Summary ─────────────────────────────────────────────────────────
        story += self._build_llm_section(styles, llm_result)
        story.append(Spacer(1, 6 * mm))

        # ── Alignment ───────────────────────────────────────────────────────────
        if ad_copy.strip():
            story += self._build_alignment_section(styles, llm_result, ad_copy)

        # ── Simulation notice ────────────────────────────────────────────────────
        if brain_result.get("simulation_mode"):
            story.append(Spacer(1, 8 * mm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
            story.append(Spacer(1, 3 * mm))
            story.append(
                Paragraph(
                    "SIMULATION MODE: TRIBE v2 model weights were not loaded. "
                    "This report demonstrates the pipeline structure only. "
                    "Replace model weights to generate clinically valid predictions.",
                    styles["notice"],
                )
            )

        doc.build(story)
        logger.info("PDF report saved: %s", out_path)
        return out_path

    # ── Cover page ─────────────────────────────────────────────────────────────
    def _build_cover(self, styles, job_id, user, brain_result):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        items = [
            Spacer(1, 20 * mm),
            Paragraph("SPIKD", styles["cover_title"]),
            Paragraph("Cortical Attention Analysis Report", styles["cover_subtitle"]),
            Spacer(1, 6 * mm),
            HRFlowable(width="100%", thickness=1, color=LIGHT_GREY),
            Spacer(1, 6 * mm),
            Paragraph(f"Job ID   : {job_id}", styles["cover_meta"]),
            Paragraph(f"Analyst  : {user}", styles["cover_meta"]),
            Paragraph(f"Generated: {ts}", styles["cover_meta"]),
            Paragraph(
                f"Duration : {brain_result.get('duration_sec', 0):.1f}s  |  "
                f"Frames   : {brain_result.get('n_frames', 0)}  |  "
                f"FPS      : {brain_result.get('fps', 0):.1f}",
                styles["cover_meta"],
            ),
            Spacer(1, 6 * mm),
            HRFlowable(width="100%", thickness=0.5, color=MID_GREY),
            Spacer(1, 4 * mm),
            Paragraph(
                "Neural data derived from TRIBE v2 cortical surface model (Meta Research). "
                "Atlas masking via Glasser HCP MMP1.0. "
                "Regions of interest: Dorsal Attention Network + Primary Visual Cortex.",
                styles["cover_note"],
            ),
        ]
        return items

    # ── Metrics table ──────────────────────────────────────────────────────────
    def _build_metrics(self, styles, brain_result):
        import numpy as np

        wf = brain_result.get("waveform", [])
        avg_eng = round(float(np.mean(wf)), 3) if wf else 0

        rows = [
            ["Metric", "Value"],
            ["Peak Engagement", f"{brain_result.get('peak_engagement_sec', 0):.2f}s"],
            ["Engagement Drop",  f"{brain_result.get('drop_timestamp_sec', 0):.2f}s"],
            ["Average Engagement (normalised)", f"{avg_eng:.3f}"],
            ["Video Duration",   f"{brain_result.get('duration_sec', 0):.1f}s"],
            ["Brain Regions",    "DAN + Primary Visual Cortex (V1)"],
            ["Atlas",            "Glasser HCP MMP1.0 (360 parcels)"],
            ["Simulation Mode",  str(brain_result.get("simulation_mode", False))],
        ]

        tbl = Table(rows, colWidths=[95 * mm, 75 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), DARK_GREY),
            ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 9),
            ("FONTSIZE",     (0, 1), (-1, -1), 8),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("TEXTCOLOR",    (0, 1), (-1, -1), BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [OFF_WHITE, WHITE]),
            ("GRID",         (0, 0), (-1, -1), 0.3, LIGHT_GREY),
            ("ALIGN",        (1, 0), (1, -1), "RIGHT"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))

        return [
            Paragraph("Key Metrics", styles["section_heading"]),
            Spacer(1, 3 * mm),
            tbl,
        ]

    # ── Waveform section ───────────────────────────────────────────────────────
    def _build_waveform_section(self, styles, waveform_png: Path, brain_result):
        img_width  = 170 * mm
        img_height = img_width * (4 / 12)  # aspect ratio from render (12x4 figsize)

        return [
            Paragraph("Attention Spike Waveform", styles["section_heading"]),
            Spacer(1, 3 * mm),
            Paragraph(
                "Normalised DAN + V1 activation averaged over masked vertices. "
                f"Peak at {brain_result.get('peak_engagement_sec', 0):.2f}s. "
                f"Engagement drop at {brain_result.get('drop_timestamp_sec', 0):.2f}s.",
                styles["body"],
            ),
            Spacer(1, 3 * mm),
            Image(str(waveform_png), width=img_width, height=img_height),
        ]

    # ── LLM narrative ──────────────────────────────────────────────────────────
    def _build_llm_section(self, styles, llm_result):
        items = [
            Paragraph("Neural Engagement Analysis", styles["section_heading"]),
            Spacer(1, 3 * mm),
            Paragraph(llm_result.get("summary", "No summary available."), styles["body"]),
        ]

        findings = llm_result.get("key_findings", [])
        if findings:
            items.append(Spacer(1, 3 * mm))
            items.append(Paragraph("Key Findings", styles["subsection_heading"]))
            for f in findings:
                items.append(Paragraph(f"- {f}", styles["bullet"]))

        return items

    # ── Alignment section ──────────────────────────────────────────────────────
    def _build_alignment_section(self, styles, llm_result, ad_copy):
        score = llm_result.get("alignment_score", "N/A")
        score_color = {
            "High":   "#2d6a2d",
            "Medium": "#7a6200",
            "Low":    "#8b0000",
        }.get(score, "#333333")

        return [
            HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY),
            Spacer(1, 4 * mm),
            Paragraph("Ad Copy Alignment Analysis", styles["section_heading"]),
            Spacer(1, 2 * mm),
            Paragraph("Ad Copy Submitted:", styles["subsection_heading"]),
            Paragraph(ad_copy, styles["quote"]),
            Spacer(1, 3 * mm),
            Paragraph(
                f'Alignment Score: <font color="{score_color}"><b>{score}</b></font>',
                styles["body"],
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                llm_result.get("alignment", "No alignment analysis available."),
                styles["body"],
            ),
        ]

    # ── Footer callback ────────────────────────────────────────────────────────
    def _page_footer(self, job_id: str):
        def draw_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MID_GREY)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            canvas.drawString(20 * mm, 12 * mm, f"SPIKD  |  Job: {job_id[:16]}  |  {ts}")
            canvas.drawRightString(
                W - 20 * mm, 12 * mm, f"Page {doc.page}"
            )
            canvas.restoreState()
        return draw_footer

    # ── Style definitions ──────────────────────────────────────────────────────
    def _build_styles(self) -> dict:
        return {
            "cover_title": ParagraphStyle(
                "cover_title",
                fontName="Helvetica-Bold", fontSize=32,
                textColor=BLACK, spaceAfter=4, alignment=TA_LEFT,
                leading=36,
            ),
            "cover_subtitle": ParagraphStyle(
                "cover_subtitle",
                fontName="Helvetica", fontSize=13,
                textColor=MID_GREY, spaceAfter=8, alignment=TA_LEFT,
            ),
            "cover_meta": ParagraphStyle(
                "cover_meta",
                fontName="Helvetica", fontSize=9,
                textColor=BLACK, spaceAfter=3, alignment=TA_LEFT,
            ),
            "cover_note": ParagraphStyle(
                "cover_note",
                fontName="Helvetica", fontSize=8,
                textColor=MID_GREY, spaceAfter=3, alignment=TA_LEFT,
                leading=12,
            ),
            "section_heading": ParagraphStyle(
                "section_heading",
                fontName="Helvetica-Bold", fontSize=11,
                textColor=BLACK, spaceAfter=2, spaceBefore=4,
                borderPad=(0, 0, 2, 0),
            ),
            "subsection_heading": ParagraphStyle(
                "subsection_heading",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=DARK_GREY, spaceAfter=2,
            ),
            "body": ParagraphStyle(
                "body",
                fontName="Helvetica", fontSize=9,
                textColor=BLACK, spaceAfter=4, leading=14,
                alignment=TA_JUSTIFY,
            ),
            "bullet": ParagraphStyle(
                "bullet",
                fontName="Helvetica", fontSize=9,
                textColor=BLACK, spaceAfter=3, leading=13,
                leftIndent=10,
            ),
            "quote": ParagraphStyle(
                "quote",
                fontName="Helvetica-Oblique", fontSize=9,
                textColor=MID_GREY, spaceAfter=4, leading=13,
                leftIndent=10, rightIndent=10,
                borderPad=4,
            ),
            "notice": ParagraphStyle(
                "notice",
                fontName="Helvetica", fontSize=8,
                textColor=MID_GREY, spaceAfter=3, leading=12,
                alignment=TA_CENTER,
            ),
        }
