import csv
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from config import Config
from modules.attendance.manager import AttendanceManager, Session, AttendanceRecord, EngagementInterval, TeachingSuggestion

class ReportExporter:
    """
    Generates CSV registers and printable HTML/PDF reports for classroom sessions.
    """
    def __init__(self, attendance_manager: AttendanceManager):
        self.attendance_mgr = attendance_manager
        self.exports_dir = Path(Config.EXPORTS_DIR)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def generate_attendance_csv(self, session_id: int) -> str:
        """Generates a CSV export for a session's attendance."""
        db = self.attendance_mgr.get_db()
        try:
            session = db.query(Session).filter(Session.id == session_id).first()
            if not session:
                raise ValueError(f"Session #{session_id} not found.")

            records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()
            
            filename = f"attendance_session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = self.exports_dir / filename

            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header metadata
                writer.writerow(["Session Name", session.name])
                writer.writerow(["Subject", session.subject])
                writer.writerow(["Started At", session.started_at.strftime("%Y-%m-%d %H:%M:%S") if session.started_at else ""])
                writer.writerow(["Duration (sec)", session.duration_seconds])
                writer.writerow(["Enrolled Students", session.total_enrolled])
                writer.writerow(["Present Count", session.present_count])
                writer.writerow(["Average Engagement (%)", session.avg_engagement])
                writer.writerow([])
                # Table headers
                writer.writerow([
                    "Student ID",
                    "Roll Number",
                    "Student Name",
                    "Status",
                    "Sustained Presence Verified",
                    "First Seen",
                    "Last Seen",
                    "Total Presence (Seconds)",
                    "Total Presence (Minutes)"
                ])

                for r in records:
                    s_name = r.student.name if r.student else "Unknown"
                    s_roll = r.student.roll_number if r.student else "-"
                    pres_min = round(r.total_presence_seconds / 60.0, 2)
                    writer.writerow([
                        r.student_id,
                        s_roll,
                        s_name,
                        r.status,
                        "Yes" if r.sustained_verified else "No",
                        r.first_seen.strftime("%H:%M:%S") if r.first_seen else "--",
                        r.last_seen.strftime("%H:%M:%S") if r.last_seen else "--",
                        round(r.total_presence_seconds, 1),
                        pres_min
                    ])

            return str(file_path)
        finally:
            db.close()

    def generate_html_report(self, session_id: int) -> str:
        """Generates a standalone, beautifully styled printable HTML report."""
        db = self.attendance_mgr.get_db()
        try:
            session = db.query(Session).filter(Session.id == session_id).first()
            if not session:
                return "<h1>Session not found</h1>"

            records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()
            intervals = db.query(EngagementInterval).filter(EngagementInterval.session_id == session_id).all()
            suggestions = db.query(TeachingSuggestion).filter(TeachingSuggestion.session_id == session_id).all()

            # Render HTML
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{session.name} - Multimodal Attendance & Engagement Report</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0A0A0A;
    color: #FFFFFF;
    margin: 0;
    padding: 30px;
  }}
  .report-container {{ max-width: 900px; margin: 0 auto; }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #C5F82A;
    padding-bottom: 20px;
    margin-bottom: 30px;
  }}
  .logo {{ font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #FFFFFF; }}
  .logo span {{ color: #C5F82A; }}
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-bottom: 30px;
  }}
  .stat-card {{
    background: #1A1D1A;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
  }}
  .stat-num {{ font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; color: #C5F82A; margin-bottom: 4px; }}
  .stat-label {{ font-size: 13px; color: #9CA3A0; text-transform: uppercase; letter-spacing: 0.5px; }}
  h2 {{ font-family: 'Space Grotesk', sans-serif; font-size: 20px; margin-top: 30px; margin-bottom: 15px; color: #FFFFFF; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; background: #121412; border-radius: 12px; overflow: hidden; }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.06); font-size: 14px; }}
  th {{ background: #1A1D1A; color: #9CA3A0; font-weight: 600; text-transform: uppercase; font-size: 12px; }}
  .badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
  }}
  .badge-present {{ background: rgba(197, 248, 42, 0.15); color: #C5F82A; border: 1px solid #C5F82A; }}
  .badge-late {{ background: rgba(245, 184, 61, 0.15); color: #F5B83D; border: 1px solid #F5B83D; }}
  .badge-absent {{ background: rgba(255, 92, 92, 0.15); color: #FF5C5C; border: 1px solid #FF5C5C; }}
  .suggestion-box {{
    background: #1A1D1A;
    border-left: 4px solid #C5F82A;
    border-radius: 0 12px 12px 0;
    padding: 16px;
    margin-bottom: 12px;
  }}
  .suggestion-time {{ font-size: 12px; color: #C5F82A; margin-bottom: 6px; font-weight: 600; }}
  .footer {{
    margin-top: 40px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #9CA3A0;
  }}
  .no-print {{ text-align: right; margin-bottom: 20px; }}
  .btn-print {{
    background: #C5F82A;
    color: #0A0A0A;
    border: none;
    padding: 10px 24px;
    border-radius: 999px;
    font-weight: 700;
    cursor: pointer;
    font-size: 14px;
  }}
  @media print {{
    body {{ background: #FFF !important; color: #000 !important; padding: 0 !important; }}
    .no-print {{ display: none !important; }}
    .stat-card, table, .suggestion-box {{ background: #F8F9FA !important; border-color: #DDD !important; color: #000 !important; }}
    .logo, h2 {{ color: #000 !important; }}
    .stat-num {{ color: #10B981 !important; }}
    .badge-present {{ background: #D1FAE5 !important; color: #065F46 !important; border-color: #10B981 !important; }}
    th {{ background: #E5E7EB !important; color: #374151 !important; }}
    td {{ color: #111827 !important; border-bottom: 1px solid #E5E7EB !important; }}
  }}
</style>
</head>
<body>
<div class="report-container">
  <div class="no-print">
    <button class="btn-print" onclick="window.print()">Print / Save as PDF</button>
  </div>
  
  <div class="header">
    <div>
      <div class="logo">Multimodal Attendance <span>& Engagement</span></div>
      <p style="color: #9CA3A0; margin-top: 4px; font-size: 14px;">Offline AI Classroom Analytics Report</p>
    </div>
    <div style="text-align: right;">
      <h3 style="margin: 0; font-size: 18px;">{session.name}</h3>
      <p style="margin: 4px 0 0; color: #9CA3A0; font-size: 14px;">Subject: {session.subject} | {session.started_at.strftime('%b %d, %Y') if session.started_at else ''}</p>
    </div>
  </div>

  <div class="meta-grid">
    <div class="stat-card">
      <div class="stat-num">{session.present_count} / {session.total_enrolled}</div>
      <div class="stat-label">Students Present</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">{session.avg_engagement}%</div>
      <div class="stat-label">Avg Engagement</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">{session.duration_seconds // 60}m {session.duration_seconds % 60}s</div>
      <div class="stat-label">Session Duration</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">{len(suggestions)}</div>
      <div class="stat-label">AI Insights Triggered</div>
    </div>
  </div>

  <h2>Attendance Register</h2>
  <table>
    <thead>
      <tr>
        <th>Roll No</th>
        <th>Student Name</th>
        <th>Status</th>
        <th>Sustained Presence</th>
        <th>First Seen</th>
        <th>Last Seen</th>
        <th>Presence Time</th>
      </tr>
    </thead>
    <tbody>
"""
            for r in records:
                s_name = r.student.name if r.student else "Unknown"
                s_roll = r.student.roll_number if r.student else "-"
                badge_class = "badge-present" if r.status in ("Present", "Late") else "badge-absent"
                if r.status == "Late":
                    badge_class = "badge-late"

                html += f"""      <tr>
        <td>{s_roll}</td>
        <td><strong>{s_name}</strong></td>
        <td><span class="badge {badge_class}">{r.status}</span></td>
        <td>{"Verified (>=70%)" if r.sustained_verified else "No"}</td>
        <td>{r.first_seen.strftime('%H:%M:%S') if r.first_seen else '--'}</td>
        <td>{r.last_seen.strftime('%H:%M:%S') if r.last_seen else '--'}</td>
        <td>{round(r.total_presence_seconds / 60.0, 1)} min</td>
      </tr>
"""

            html += """    </tbody>
  </table>

  <h2>AI Teaching Insights & Interventions</h2>
"""
            if not suggestions:
                html += '<p style="color: #9CA3A0;">No critical attention drops occurred during this session. Engagement remained balanced.</p>'
            else:
                for s in suggestions:
                    html += f"""  <div class="suggestion-box">
    <div class="suggestion-time">{s.timestamp.strftime('%H:%M:%S')} &bull; {s.trigger_type.upper()} &bull; Source: {s.source} ({s.model_name})</div>
    <div style="font-size: 14px; line-height: 1.5;">{s.suggestion_text}</div>
  </div>
"""

            html += f"""  <div class="footer">
    <div>Developed by <strong>Ashutosh Pandey</strong> &bull; LNCT College, CSE AIML 3rd Year</div>
    <div>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &bull; Multimodal Attendance Agent</div>
  </div>
</div>
</body>
</html>
"""
            # Also save report to disk
            report_file = self.exports_dir / f"session_{session_id}_report.html"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(html)

            return html
        finally:
            db.close()
