"""Report generation service producing CSV and academic PDF summaries."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import pandas as pd
from datetime import datetime

from config.settings import REPORTS_DIR, DEFAULT_CONFIG
from database.repository import repo
from utils.timestamps import format_seconds_to_mmss
from utils.logging_config import logger


class ReportService:
    """Generates tabular CSV dataframes and multi-page academic PDF reports."""

    def generate_attendance_csv(self, session_id: int) -> str:
        """Generate formatted CSV string for session attendance."""
        records = repo.get_attendance_for_session(session_id)
        if not records:
            return "No attendance records found."

        df = pd.DataFrame(records)
        columns_to_keep = [
            "student_id", "student_name", "track_id", "status",
            "first_seen_timestamp", "last_seen_timestamp", "total_visible_seconds",
            "presence_ratio", "verification_type", "notes"
        ]
        available_cols = [c for c in columns_to_keep if c in df.columns]
        export_df = df[available_cols].copy()
        export_df["presence_percentage"] = (export_df["presence_ratio"] * 100).round(1).astype(str) + "%"

        return export_df.to_csv(index=False)

    def generate_fused_metrics_csv(self, session_id: int) -> str:
        """Generate formatted CSV string for windowed multimodal metrics."""
        metrics = repo.get_fused_metrics_for_session(session_id)
        if not metrics:
            return "No metrics available."

        df = pd.DataFrame(metrics)
        columns_to_keep = [
            "window_start_sec", "window_end_sec", "visual_activity_index",
            "speech_activity_index", "composite_activity_score",
            "active_participants_count", "data_quality_index", "signal_availability"
        ]
        available_cols = [c for c in columns_to_keep if c in df.columns]
        return df[available_cols].to_csv(index=False)

    def generate_session_pdf(self, session_id: int, output_path: Optional[Path] = None) -> Path:
        """Build a professional, calm academic PDF report using ReportLab."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        if output_path is None:
            output_path = REPORTS_DIR / f"Session_{session_id}_Academic_Report.pdf"

        session = repo.get_session(session_id)
        attendance = repo.get_attendance_for_session(session_id)
        fused = repo.get_fused_metrics_for_session(session_id)
        recommendations = repo.get_recommendations_for_session(session_id)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1e293b"),
            fontName="Helvetica-Bold"
        )
        heading2_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748b"),
            fontName="Helvetica-Oblique"
        )

        elements = []

        # Header Title & Date
        session_name = session.get("session_name", f"Session #{session_id}") if session else f"Session #{session_id}"
        course_name = session.get("course_name", "N/A") if session else "N/A"
        date_str = session.get("date", datetime.now().strftime("%Y-%m-%d")) if session else ""
        duration_sec = session.get("duration_seconds", 0.0) if session else 0.0

        elements.append(Paragraph("AURORA — Multimodal Classroom Intelligence Report", title_style))
        elements.append(Paragraph(f"<b>Session:</b> {session_name} | <b>Course:</b> {course_name} | <b>Date:</b> {date_str} | <b>Duration:</b> {format_seconds_to_mmss(duration_sec)}", body_style))
        elements.append(Paragraph("<i>Offline Automated Attendance, Observable Engagement Trends & Pedagogical Insights</i>", disclaimer_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

        # 1. Summary Statistics Table
        total_p = sum(1 for a in attendance if a["status"] == "Present")
        total_l = sum(1 for a in attendance if a["status"] == "Late")
        total_a = sum(1 for a in attendance if a["status"] == "Absent")
        total_rev = sum(1 for a in attendance if a["status"] == "Needs Review")
        total_tracks = len(attendance)
        att_rate = round(((total_p + total_l) / total_tracks * 100), 1) if total_tracks > 0 else 0.0

        avg_act = 0.0
        if fused:
            avg_act = round(sum(f["composite_activity_score"] for f in fused) / len(fused), 2)

        summary_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Participants Tracked", str(total_tracks), "Attendance Rate", f"{att_rate}%"],
            ["Present", str(total_p), "Late Arrivals", str(total_l)],
            ["Absent / Fleet Visits", str(total_a), "Average Activity Index", f"{avg_act} / 1.0"]
        ]
        t_summary = Table(summary_data, colWidths=[140, 110, 140, 110])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(t_summary)
        elements.append(Spacer(1, 12))

        # 2. Attendance Roster Section
        elements.append(Paragraph("1. Attendance Roster & Sustained Presence", heading2_style))
        if attendance:
            att_table_data = [["Student / Track", "Name", "Visible Time", "Ratio", "Status", "Verification"]]
            for r in attendance[:25]:  # Show up to 25 records
                sid = r.get("student_id") or f"Track #{r['track_id']}"
                sname = r.get("student_name") or "Anonymous"
                vis_str = format_seconds_to_mmss(r.get("total_visible_seconds", 0.0))
                ratio_str = f"{int(r.get('presence_ratio', 0.0) * 100)}%"
                status = r.get("status", "Absent")
                verif = r.get("verification_type", "anonymous")
                att_table_data.append([sid, sname, vis_str, ratio_str, status, verif])

            t_att = Table(att_table_data, colWidths=[100, 120, 75, 55, 75, 75])
            t_att.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_att)
        else:
            elements.append(Paragraph("No participants recorded for this session.", body_style))
        elements.append(Spacer(1, 12))

        # 3. AI Pedagogical Recommendations Section
        elements.append(Paragraph("2. Advisory Teaching Insights", heading2_style))
        if recommendations:
            for rec in recommendations:
                cat = rec.get("category", "Insight")
                sug = rec.get("suggestion", "")
                evid = rec.get("evidence", "")
                lim = rec.get("limitations", "")
                conf = int(rec.get("confidence", 0.8) * 100)

                rec_text = f"<b>[{cat}] - Confidence: {conf}%</b><br/>"
                rec_text += f"<b>Suggestion:</b> {sug}<br/>"
                rec_text += f"<b>Observed Evidence:</b> {evid}<br/>"
                rec_text += f"<i><b>Limitations:</b> {lim}</i>"

                elements.append(Paragraph(rec_text, body_style))
                elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph("No specific intervention triggers observed during this session.", body_style))
        elements.append(Spacer(1, 14))

        # 4. Privacy & Scientific Disclaimer
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
        elements.append(Paragraph(
            "<b>Scientific Disclaimer & Privacy Notice:</b> All calculations were processed locally without transmitting audio, video, or biometric signatures to third-party or cloud servers. Observable activity metrics represent physical movement, head alignment, and acoustic levels. They do NOT measure cognitive attention, understanding, or emotional engagement. Recommendations are advisory and must not be used for punitive or grading decisions.",
            disclaimer_style
        ))

        doc.build(elements)
        logger.info(f"Generated PDF session report at {output_path}")
        return output_path


report_service = ReportService()
