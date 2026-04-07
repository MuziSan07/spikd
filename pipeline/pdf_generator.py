"""
pipeline/pdf_generator.py - with logo, recommendations, peak frame, target audience
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.lib.styles import ParagraphStyle

logger = logging.getLogger("spikd.pdf")

BLACK    = colors.HexColor("#000000")
MID_GREY = colors.HexColor("#2a2a2a")
DIM_WHT  = colors.HexColor("#666666")
ROW_ALT  = colors.HexColor("#080808")
WHITE    = colors.HexColor("#ffffff")
ACCENT   = colors.HexColor("#cccccc")

MONO      = "Courier"
MONO_BOLD = "Courier-Bold"
W, H      = A4


class PDFGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def generate(self, job_id, job_dir, brain_result, llm_result,
                 ad_copy, target_audience, user, logo_path=None, peak_frame_path=None):
        out_path = job_dir / "report.pdf"
        S = self._styles()

        doc = BaseDocTemplate(
            str(out_path), pagesize=A4,
            leftMargin=16*mm, rightMargin=16*mm,
            topMargin=14*mm, bottomMargin=14*mm,
        )
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
        doc.addPageTemplates([PageTemplate(
            id="main", frames=[frame],
            onPage=self._page_chrome(job_id, logo_path),
        )])

        story = []
        story += self._section_header(S, job_id, user, target_audience, logo_path)
        story += self._section_graph(S, doc, brain_result, peak_frame_path)
        story += self._section_stats(S, doc, brain_result)
        story += self._section_verdict(S, llm_result)
        story += self._section_recommendations(S, llm_result)

        if ad_copy.strip():
            story.append(PageBreak())
            story += self._alignment_page(S, llm_result, ad_copy, target_audience)

        doc.build(story)
        return out_path

    def _section_header(self, S, job_id, user, target_audience, logo_path):
        ts = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
        items = []

        if logo_path and Path(logo_path).exists():
            try:
                logo_img = Image(logo_path, width=40*mm, height=14*mm)
                logo_img.hAlign = "RIGHT"
                header_table = Table(
                    [[Paragraph("NEURAL ATTENTION REPORT", S["report_title"]), logo_img]],
                    colWidths=["75%", "25%"],
                )
                header_table.setStyle(TableStyle([
                    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                    ("BACKGROUND",   (0,0), (-1,-1), BLACK),
                    ("LEFTPADDING",  (0,0), (-1,-1), 0),
                    ("RIGHTPADDING", (0,0), (-1,-1), 0),
                    ("TOPPADDING",   (0,0), (-1,-1), 0),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 0),
                ]))
                items.append(header_table)
            except Exception:
                items.append(Paragraph("NEURAL ATTENTION REPORT", S["report_title"]))
        else:
            items.append(Paragraph("NEURAL ATTENTION REPORT", S["report_title"]))

        meta_line = f"JOB: {job_id[:24].upper()}   |   ANALYST: {user.upper()}   |   {ts}"
        if target_audience:
            meta_line += f"   |   AUDIENCE: {target_audience.upper()}"

        items += [
            Spacer(1, 1*mm),
            Paragraph(meta_line, S["header_sub"]),
            Spacer(1, 2*mm),
            HRFlowable(width="100%", thickness=1.0, color=WHITE),
            Spacer(1, 1.5*mm),
            Paragraph(
                "SOURCE: TRIBE v2 (META RESEARCH)   |   ATLAS: GLASSER HCP MMP1.0   |   REGIONS: DAN + V1",
                S["header_meta"],
            ),
            Spacer(1, 4*mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4*mm),
        ]
        return items

    def _section_graph(self, S, doc, brain_result, peak_frame_path):
        items = [
            Paragraph("ATTENTION WAVEFORM  //  CORTICAL SURFACE OUTPUT", S["section_label"]),
            Spacer(1, 2*mm),
        ]

        waveform_png = Path(brain_result.get("waveform_png_path", ""))

        if peak_frame_path and Path(peak_frame_path).exists() and waveform_png.exists():
            img_w = doc.width * 0.78
            img_h = img_w * (4/12)
            peak_img_w = doc.width * 0.20
            peak_img_h = peak_img_w * (9/16)
            graph_table = Table(
                [[
                    Image(str(waveform_png), width=img_w, height=img_h),
                    Table([[
                        Paragraph("PEAK FRAME", S["notice"]),
                        Image(str(peak_frame_path), width=peak_img_w, height=peak_img_h),
                        Paragraph(f"@ {brain_result.get('peak_engagement_sec', 0):.2f}s", S["notice"]),
                    ]], colWidths=[peak_img_w]),
                ]],
                colWidths=[img_w, doc.width * 0.22],
            )
            graph_table.setStyle(TableStyle([
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("BACKGROUND",   (0,0), (-1,-1), BLACK),
                ("LEFTPADDING",  (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 4),
                ("TOPPADDING",   (0,0), (-1,-1), 0),
                ("BOTTOMPADDING",(0,0), (-1,-1), 0),
            ]))
            items.append(graph_table)
        elif waveform_png.exists():
            img_w = doc.width
            img_h = img_w * (4/12)
            items.append(Image(str(waveform_png), width=img_w, height=img_h))
        else:
            items.append(Paragraph("[ WAVEFORM NOT AVAILABLE ]", S["notice"]))

        items += [Spacer(1, 4*mm), HRFlowable(width="100%", thickness=0.3, color=MID_GREY), Spacer(1, 4*mm)]
        return items

    def _section_stats(self, S, doc, brain_result):
        wf           = brain_result.get("waveform", [])
        peak_sec     = brain_result.get("peak_engagement_sec", 0.0)
        drop_sec     = brain_result.get("drop_timestamp_sec", 0.0)
        duration_sec = brain_result.get("duration_sec", 1.0)
        fps          = brain_result.get("fps", 30.0)
        n_frames     = brain_result.get("n_frames", 0)

        hook_strength = decay_rate = 0.0
        if wf:
            arr   = np.array(wf)
            cutoff = max(1, int(len(arr) * 0.2))
            hook_strength = round(float(arr[:cutoff].mean()), 4)
            peak_idx = int(np.argmax(arr))
            post = arr[peak_idx:]
            if len(post) > 1 and fps > 0:
                t = np.arange(len(post)) / fps
                decay_rate = round(float(np.polyfit(t, post, 1)[0]), 5)

        def fmt(sec): return f"{int(sec)//60:02d}:{int(sec)%60:02d}"

        rows = [
            ["METRIC", "VALUE", "INTERPRETATION"],
            ["Peak Engagement",       fmt(peak_sec),          f"Highest cortical activation at {peak_sec:.1f}s"],
            ["Hook Strength (0-1)",   f"{hook_strength:.4f}", "Mean activation across first 20% of video"],
            ["Attention Decay Rate",  f"{decay_rate:+.5f}/sec","Slope post-peak. Negative = losing attention"],
            ["Engagement Drop At",    fmt(drop_sec),          f"Sharpest attention decline at {drop_sec:.1f}s"],
            ["Total Duration",        fmt(duration_sec),      f"{duration_sec:.1f} seconds at {fps:.1f} fps"],
        ]

        col_w = [doc.width*0.35, doc.width*0.19, doc.width*0.46]
        tbl   = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  MID_GREY),
            ("TEXTCOLOR",      (0,0), (-1,0),  WHITE),
            ("FONTNAME",       (0,0), (-1,0),  MONO_BOLD),
            ("FONTSIZE",       (0,0), (-1,0),  7),
            ("BACKGROUND",     (0,1), (-1,-1), BLACK),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [BLACK, ROW_ALT]),
            ("TEXTCOLOR",      (0,1), (-1,-1), WHITE),
            ("FONTNAME",       (0,1), (-1,-1), MONO),
            ("FONTSIZE",       (0,1), (-1,-1), 8),
            ("FONTNAME",       (1,1), (1,-1),  MONO_BOLD),
            ("FONTSIZE",       (1,1), (1,-1),  8.5),
            ("TEXTCOLOR",      (2,1), (2,-1),  DIM_WHT),
            ("FONTSIZE",       (2,1), (2,-1),  7),
            ("GRID",           (0,0), (-1,-1), 0.4, MID_GREY),
            ("TOPPADDING",     (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
            ("LEFTPADDING",    (0,0), (-1,-1), 6),
            ("RIGHTPADDING",   (0,0), (-1,-1), 6),
            ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ]))

        return [
            Paragraph("ENGAGEMENT METRICS", S["section_label"]),
            Spacer(1, 2*mm), tbl,
            Spacer(1, 4*mm), HRFlowable(width="100%", thickness=0.3, color=MID_GREY), Spacer(1, 4*mm),
        ]

    def _section_verdict(self, S, llm_result):
        items = [Paragraph("KEY OBSERVATIONS  //  AI ANALYSIS", S["section_label"]), Spacer(1, 3*mm)]
        observations = llm_result.get("observations", [])
        summary      = llm_result.get("summary", "")

        if observations:
            for line in observations:
                style = S["obs_alert"] if "[ALERT]" in line else S["obs_line"]
                items.append(Paragraph(line, style))
                items.append(Spacer(1, 1.5*mm))
        elif summary:
            items.append(Paragraph(f"00:00 [SUMMARY]: {summary}", S["obs_line"]))
        else:
            items.append(Paragraph("[ NO OBSERVATIONS GENERATED ]", S["notice"]))

        findings = llm_result.get("key_findings", [])
        if findings:
            items += [Spacer(1, 3*mm), Paragraph("SUPPORTING FINDINGS", S["section_label"]), Spacer(1, 2*mm)]
            for f in findings:
                items.append(Paragraph(f"  --  {f}", S["finding"]))
                items.append(Spacer(1, 1*mm))

        return items

    def _section_recommendations(self, S, llm_result):
        recs = llm_result.get("recommendations", [])
        if not recs:
            return []

        items = [
            Spacer(1, 4*mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4*mm),
            Paragraph("3 RECOMMENDED EDITS", S["section_label"]),
            Spacer(1, 3*mm),
        ]
        for i, rec in enumerate(recs[:3], 1):
            items.append(Paragraph(f"  {i}.  {rec}", S["obs_line"]))
            items.append(Spacer(1, 2*mm))

        return items

    def _alignment_page(self, S, llm_result, ad_copy, target_audience=""):
        score = llm_result.get("alignment_score", "N/A").upper()
        items = [
            Paragraph("NEURAL ATTENTION REPORT  //  AD COPY ALIGNMENT", S["report_title"]),
            Spacer(1, 2*mm),
            HRFlowable(width="100%", thickness=1.0, color=WHITE),
            Spacer(1, 5*mm),
        ]

        if target_audience:
            items += [
                Paragraph("TARGET AUDIENCE", S["section_label"]),
                Spacer(1, 2*mm),
                Paragraph(target_audience, S["obs_line"]),
                Spacer(1, 5*mm),
                HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
                Spacer(1, 4*mm),
            ]

        items += [
            Paragraph("AD COPY SUBMITTED", S["section_label"]),
            Spacer(1, 2*mm),
            Paragraph(ad_copy, S["quote"]),
            Spacer(1, 5*mm),
            HRFlowable(width="100%", thickness=0.3, color=MID_GREY),
            Spacer(1, 4*mm),
            Paragraph(f"ALIGNMENT VERDICT:  {score}", S["alignment_score"]),
            Spacer(1, 4*mm),
            Paragraph("ALIGNMENT OBSERVATIONS", S["section_label"]),
            Spacer(1, 2*mm),
            Paragraph(llm_result.get("alignment", "No alignment analysis available."), S["obs_line"]),
        ]
        return items

    def _page_chrome(self, job_id, logo_path=None):
        def on_page(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(BLACK)
            canvas.rect(0, 0, W, H, fill=1, stroke=0)
            canvas.setStrokeColor(MID_GREY)
            canvas.setLineWidth(0.3)
            canvas.line(16*mm, 11*mm, W-16*mm, 11*mm)
            canvas.setFont(MONO, 6)
            canvas.setFillColor(DIM_WHT)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            canvas.drawString(16*mm, 7.5*mm, f"NEURAL ATTENTION TELEMETRY  |  {job_id[:32].upper()}  |  {ts}")
            canvas.drawRightString(W-16*mm, 7.5*mm, f"PAGE {doc.page}")
            canvas.restoreState()
        return on_page

    def _styles(self):
        return {
            "report_title":    ParagraphStyle("report_title",    fontName=MONO_BOLD, fontSize=11, textColor=WHITE,   leading=16, alignment=TA_LEFT,   backColor=BLACK, letterSpacing=1.2),
            "header_sub":      ParagraphStyle("header_sub",      fontName=MONO,      fontSize=7,  textColor=DIM_WHT, leading=11, alignment=TA_LEFT,   backColor=BLACK),
            "header_meta":     ParagraphStyle("header_meta",     fontName=MONO,      fontSize=6.5,textColor=DIM_WHT, leading=10, alignment=TA_LEFT,   backColor=BLACK),
            "section_label":   ParagraphStyle("section_label",   fontName=MONO_BOLD, fontSize=7.5,textColor=WHITE,   leading=12, alignment=TA_LEFT,   backColor=BLACK, letterSpacing=1.5),
            "obs_line":        ParagraphStyle("obs_line",        fontName=MONO,      fontSize=8.5,textColor=WHITE,   leading=14, alignment=TA_LEFT,   backColor=BLACK, leftIndent=4),
            "obs_alert":       ParagraphStyle("obs_alert",       fontName=MONO_BOLD, fontSize=8.5,textColor=WHITE,   leading=14, alignment=TA_LEFT,   backColor=BLACK, leftIndent=4),
            "finding":         ParagraphStyle("finding",         fontName=MONO,      fontSize=8,  textColor=DIM_WHT, leading=13, alignment=TA_LEFT,   backColor=BLACK, leftIndent=10),
            "quote":           ParagraphStyle("quote",           fontName=MONO,      fontSize=8,  textColor=DIM_WHT, leading=13, alignment=TA_LEFT,   backColor=BLACK, leftIndent=8),
            "alignment_score": ParagraphStyle("alignment_score", fontName=MONO_BOLD, fontSize=12, textColor=WHITE,   leading=18, alignment=TA_LEFT,   backColor=BLACK, letterSpacing=1),
            "notice":          ParagraphStyle("notice",          fontName=MONO,      fontSize=6.5,textColor=DIM_WHT, leading=10, alignment=TA_CENTER, backColor=BLACK),
        }
