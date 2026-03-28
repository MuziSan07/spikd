"""
pipeline/pdf_generator.py

4-Section Clinical Lab Report Layout:

  1. HEADER     — SPIKD NEURAL-ATTENTION TELEMETRY REPORT (monospace bold)
  2. GRAPH      — TRIBE v2 Attention Amplitude waveform (full width)
  3. STATS TABLE — Peak Resonance / Initial Hook Strength / Attention Decay Rate
  4. VERDICT    — Llama 3.2 formatted as clinical Observations
                  e.g.  0:04 [ALERT]: Text Overload

Design: pitch black background, Courier monospace throughout,
        white line graph, no fill, no gradients, no colour.
        Hospital lab result aesthetic — not a marketing deck.
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
    Table,
    TableStyle,
)
from reportlab.lib.styles import ParagraphStyle

logger = logging.getLogger("spikd.pdf")

# ── Palette — pitch black only ────────────────────────────────────────────────
BLACK    = colors.HexColor("#000000")
MID_GREY = colors.HexColor("#2a2a2a")
DIM_WHT  = colors.HexColor("#666666")
ROW_ALT  = colors.HexColor("#080808")
WHITE    = colors.HexColor("#ffffff")

MONO      = "Courier"
MONO_BOLD = "Courier-Bold"
W, H      = A4


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
        out_path = job_dir / "report.pdf"
        S = self._styles()

        doc = BaseDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=16 * mm,
            rightMargin=16 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
        )
        frame = Frame(
            doc.leftMargin, doc.bottomMargin,
            doc.width, doc.height,
            id="main",
        )
        doc.addPageTemplates([
            PageTemplate(
                id="main",
                frames=[frame],
                onPage=self._page_chrome(job_id),
            )
        ])

        story = []
        story += self._section_1_header(S, job_id, user)
        story += self._section_2_graph(S, doc, brain_result)
        story += self._section_3_stats(S, doc, brain_result)
        story += self._section_4_verdict(S, llm_result)

        if ad_copy.strip():
            story.append(PageBreak())
            story += self._alignment_page(S, llm_result, ad_copy)

        if brain_result.get("simulation_mode"):
            story.append(Spacer(1, 5 * mm))
            story.append(HRFlowable(width="100%", thickness=0.3, color=MID_GREY))
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                "[ SIMULATION MODE — TRIBE v2 WEIGHTS NOT LOADED — "
                "SYNTHETIC DATA ONLY — NOT FOR CLINICAL USE ]",
                S["notice"],
            ))

        doc.build(story)
        logger.info("PDF written: %s", out_path)
        return out_path

    # ── SECTION 1 — HEADER ────────────────────────────────────────────────────
    def _section_1_header(self, S, job_id, user):
        ts = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
        return [
            Paragraph(
                "SPIKD NEURAL-ATTENTION TELEMETRY REPORT",
                S["report_title"],
            ),
            Spacer(1, 1 * mm),
            Paragraph(
                f"JOB: {job_id[:24].upper()}   |   ANALYST: {user.upper()}   |   {ts}",
                S["header_sub"],
            ),
            Spacer(1, 2 * mm),
            HRFlowable(width="100%", thickness=1.0, color=WHITE),
            Spacer(1, 1.5 * mm),
            Paragraph(
                "SOURCE: TRIBE v2 (META RESEARCH)   |   "
                "ATLAS: GLASSER HCP MMP1.0   |   "
                "REGIONS: DORSAL ATTENTION NETWORK + PRIMARY VISUAL CORTEX (V1)",
                S["header_meta"],
            ),
            Spacer(1, 4 * mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4 * mm),
        ]

    # ── SECTION 2 — GRAPH ─────────────────────────────────────────────────────
    def _section_2_graph(self, S, doc, brain_result):
        items = [
            Paragraph(
                "ATTENTION AMPLITUDE  //  TRIBE v2 CORTICAL SURFACE OUTPUT",
                S["section_label"],
            ),
            Spacer(1, 2 * mm),
        ]

        waveform_png = Path(brain_result.get("waveform_png_path", ""))
        if waveform_png.exists():
            img_w = doc.width
            img_h = img_w * (4 / 12)
            items.append(Image(str(waveform_png), width=img_w, height=img_h))
        else:
            items.append(Paragraph("[ WAVEFORM RENDER NOT AVAILABLE ]", S["notice"]))

        items += [
            Spacer(1, 4 * mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4 * mm),
        ]
        return items

    # ── SECTION 3 — STATS TABLE ───────────────────────────────────────────────
    def _section_3_stats(self, S, doc, brain_result):
        wf           = brain_result.get("waveform", [])
        peak_sec     = brain_result.get("peak_engagement_sec", 0.0)
        drop_sec     = brain_result.get("drop_timestamp_sec", 0.0)
        duration_sec = brain_result.get("duration_sec", 1.0)
        fps          = brain_result.get("fps", 30.0)
        n_frames     = brain_result.get("n_frames", 0)

        hook_strength = 0.0
        decay_rate    = 0.0
        if wf:
            arr      = np.array(wf)
            cutoff   = max(1, int(len(arr) * 0.2))
            hook_strength = round(float(arr[:cutoff].mean()), 4)
            peak_idx = int(np.argmax(arr))
            post     = arr[peak_idx:]
            if len(post) > 1 and fps > 0:
                t    = np.arange(len(post)) / fps
                decay_rate = round(float(np.polyfit(t, post, 1)[0]), 5)

        def fmt(sec):
            return f"{int(sec)//60:02d}:{int(sec)%60:02d}"

        rows = [
            ["METRIC", "VALUE", "INTERPRETATION"],
            [
                "Peak Resonance",
                fmt(peak_sec),
                f"Highest DAN+V1 activation — frame {int(peak_sec * fps)} of {n_frames}",
            ],
            [
                "Initial Hook Strength (0-1)",
                f"{hook_strength:.4f}",
                "Mean activation across first 20% of video duration",
            ],
            [
                "Attention Decay Rate",
                f"{decay_rate:+.5f}/sec",
                "Linear slope post-peak. Negative = audience losing attention",
            ],
            [
                "Engagement Drop At",
                fmt(drop_sec),
                "Timestamp of sharpest post-peak gradient descent",
            ],
            [
                "Total Duration",
                fmt(duration_sec),
                f"{n_frames} frames at {fps:.1f} fps",
            ],
        ]

        col_w = [doc.width * 0.35, doc.width * 0.19, doc.width * 0.46]
        tbl   = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle([
            # Header
            ("BACKGROUND",     (0, 0), (-1, 0),  MID_GREY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",       (0, 0), (-1, 0),  MONO_BOLD),
            ("FONTSIZE",       (0, 0), (-1, 0),  7),
            # Data rows
            ("BACKGROUND",     (0, 1), (-1, -1), BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLACK, ROW_ALT]),
            ("TEXTCOLOR",      (0, 1), (-1, -1), WHITE),
            ("FONTNAME",       (0, 1), (-1, -1), MONO),
            ("FONTSIZE",       (0, 1), (-1, -1), 8),
            # Value column — bold
            ("FONTNAME",       (1, 1), (1, -1),  MONO_BOLD),
            ("FONTSIZE",       (1, 1), (1, -1),  8.5),
            # Interpretation column — dimmed
            ("TEXTCOLOR",      (2, 1), (2, -1),  DIM_WHT),
            ("FONTSIZE",       (2, 1), (2, -1),  7),
            # Grid
            ("GRID",           (0, 0), (-1, -1), 0.4, MID_GREY),
            # Padding
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ]))

        return [
            Paragraph("SIGNAL METRICS", S["section_label"]),
            Spacer(1, 2 * mm),
            tbl,
            Spacer(1, 4 * mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4 * mm),
        ]

    # ── SECTION 4 — VERDICT ───────────────────────────────────────────────────
    def _section_4_verdict(self, S, llm_result):
        items = [
            Paragraph(
                "CLINICAL OBSERVATIONS  //  LLAMA 3.2 VERDICT",
                S["section_label"],
            ),
            Spacer(1, 3 * mm),
        ]

        observations = llm_result.get("observations", [])
        summary      = llm_result.get("summary", "")

        if observations:
            for line in observations:
                style = S["obs_alert"] if "[ALERT]" in line else S["obs_line"]
                items.append(Paragraph(line, style))
                items.append(Spacer(1, 1.5 * mm))
        elif summary:
            items.append(Paragraph(f"00:00 [SUMMARY]: {summary}", S["obs_line"]))
        else:
            items.append(Paragraph(
                "[ NO OBSERVATIONS GENERATED — CHECK LLM SERVICE ]",
                S["notice"],
            ))

        findings = llm_result.get("key_findings", [])
        if findings:
            items += [
                Spacer(1, 3 * mm),
                Paragraph("SUPPORTING FINDINGS", S["section_label"]),
                Spacer(1, 2 * mm),
            ]
            for f in findings:
                items.append(Paragraph(f"  --  {f}", S["finding"]))
                items.append(Spacer(1, 1 * mm))

        return items

    # ── ALIGNMENT PAGE ────────────────────────────────────────────────────────
    def _alignment_page(self, S, llm_result, ad_copy):
        score = llm_result.get("alignment_score", "N/A").upper()
        ts    = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
        return [
            Paragraph(
                "SPIKD NEURAL-ATTENTION TELEMETRY REPORT  //  AD COPY ALIGNMENT",
                S["report_title"],
            ),
            Spacer(1, 2 * mm),
            HRFlowable(width="100%", thickness=1.0, color=WHITE),
            Spacer(1, 5 * mm),
            Paragraph("AD COPY SUBMITTED", S["section_label"]),
            Spacer(1, 2 * mm),
            Paragraph(ad_copy, S["quote"]),
            Spacer(1, 5 * mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4 * mm),
            Paragraph(f"ALIGNMENT VERDICT:  {score}", S["alignment_score"]),
            Spacer(1, 4 * mm),
            Paragraph("ALIGNMENT OBSERVATIONS", S["section_label"]),
            Spacer(1, 2 * mm),
            Paragraph(
                llm_result.get("alignment", "No alignment analysis available."),
                S["obs_line"],
            ),
        ]

    # ── PAGE CHROME ───────────────────────────────────────────────────────────
    def _page_chrome(self, job_id: str):
        def on_page(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(BLACK)
            canvas.rect(0, 0, W, H, fill=1, stroke=0)
            canvas.setStrokeColor(MID_GREY)
            canvas.setLineWidth(0.3)
            canvas.line(16 * mm, 11 * mm, W - 16 * mm, 11 * mm)
            canvas.setFont(MONO, 6)
            canvas.setFillColor(DIM_WHT)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            canvas.drawString(
                16 * mm, 7.5 * mm,
                f"SPIKD TELEMETRY  |  {job_id[:32].upper()}  |  {ts}",
            )
            canvas.drawRightString(W - 16 * mm, 7.5 * mm, f"PAGE {doc.page}")
            canvas.restoreState()
        return on_page

    # ── STYLES ────────────────────────────────────────────────────────────────
    def _styles(self) -> dict:
        return {
            "report_title": ParagraphStyle(
                "report_title",
                fontName=MONO_BOLD, fontSize=11,
                textColor=WHITE, leading=16,
                alignment=TA_LEFT, backColor=BLACK,
                letterSpacing=1.2,
            ),
            "header_sub": ParagraphStyle(
                "header_sub",
                fontName=MONO, fontSize=7,
                textColor=DIM_WHT, leading=11,
                alignment=TA_LEFT, backColor=BLACK,
            ),
            "header_meta": ParagraphStyle(
                "header_meta",
                fontName=MONO, fontSize=6.5,
                textColor=DIM_WHT, leading=10,
                alignment=TA_LEFT, backColor=BLACK,
            ),
            "section_label": ParagraphStyle(
                "section_label",
                fontName=MONO_BOLD, fontSize=7.5,
                textColor=WHITE, leading=12,
                alignment=TA_LEFT, backColor=BLACK,
                letterSpacing=1.5,
            ),
            "obs_line": ParagraphStyle(
                "obs_line",
                fontName=MONO, fontSize=8.5,
                textColor=WHITE, leading=14,
                alignment=TA_LEFT, backColor=BLACK,
                leftIndent=4,
            ),
            "obs_alert": ParagraphStyle(
                "obs_alert",
                fontName=MONO_BOLD, fontSize=8.5,
                textColor=WHITE, leading=14,
                alignment=TA_LEFT, backColor=BLACK,
                leftIndent=4,
            ),
            "finding": ParagraphStyle(
                "finding",
                fontName=MONO, fontSize=8,
                textColor=DIM_WHT, leading=13,
                alignment=TA_LEFT, backColor=BLACK,
                leftIndent=10,
            ),
            "quote": ParagraphStyle(
                "quote",
                fontName=MONO, fontSize=8,
                textColor=DIM_WHT, leading=13,
                leftIndent=8, alignment=TA_LEFT, backColor=BLACK,
            ),
            "alignment_score": ParagraphStyle(
                "alignment_score",
                fontName=MONO_BOLD, fontSize=12,
                textColor=WHITE, leading=18,
                alignment=TA_LEFT, backColor=BLACK,
                letterSpacing=1,
            ),
            "notice": ParagraphStyle(
                "notice",
                fontName=MONO, fontSize=6.5,
                textColor=DIM_WHT, leading=10,
                alignment=TA_CENTER, backColor=BLACK,
            ),
        }