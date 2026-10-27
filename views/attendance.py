"""Attendance Management page displaying sustained-presence rosters, manual overrides, reassignment, and LMS export options in AURORA."""

import streamlit as st
import pandas as pd
import json
from typing import Optional
from pathlib import Path

from database.repository import repo
from services.attendance_service import attendance_service
from services.report_service import report_service
from utils.timestamps import format_seconds_to_mmss
from ui.theme import render_aurora_hero


def render_attendance(selected_session_id: Optional[int] = None) -> None:
    render_aurora_hero(
        title="ATTENDANCE DESK",
        subtitle="Sustained-presence roster verification, identity matching, manual instructor overrides, and multi-format exports",
        badge="SUSTAINED PRESENCE VERIFICATION"
    )

    sessions = repo.list_sessions()
    if not sessions:
        st.info("No recorded sessions found. Go to **Video Processing** to ingest or generate a classroom session.")
        return

    sess_options = {s["id"]: f"Session #{s['id']} • {s['session_name']} ({s['date']})" for s in sessions}
    active_sid = st.selectbox(
        "Select Classroom Session:",
        options=list(sess_options.keys()),
        index=0 if not selected_session_id or selected_session_id not in sess_options else list(sess_options.keys()).index(selected_session_id),
        format_func=lambda sid: sess_options[sid],
        key="attendance_session_select"
    )

    session = repo.get_session(active_sid)
    records = repo.get_attendance_for_session(active_sid)

    if not records:
        st.warning("No attendance records computed for this session yet.")
        return

    # Metrics Summary Bar
    p_cnt = sum(1 for r in records if r["status"] == "Present")
    l_cnt = sum(1 for r in records if r["status"] == "Late")
    a_cnt = sum(1 for r in records if r["status"] == "Absent")
    rev_cnt = sum(1 for r in records if r["status"] == "Needs Review")
    rate = round(((p_cnt + l_cnt) / len(records)) * 100, 1)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Present", p_cnt, delta="Sustained presence")
    m2.metric("Late Arrivals", l_cnt, delta="After late cutoff")
    m3.metric("Needs Review", rev_cnt, delta="Transient / partial")
    m4.metric("Absent", a_cnt, delta="< threshold")
    m5.metric("Attendance Rate", f"{rate}%")

    st.markdown("<div style='margin: 14px 0; height: 1px; background: rgba(255,255,255,0.07);'></div>", unsafe_allow_html=True)

    att_tab1, att_tab2, att_tab3, att_tab4 = st.tabs([
        "📋 Attendance Roster",
        "✏️ Manual Overrides & Reassignment",
        "➕ Quick Student Enrollment",
        "📥 Official Reports & LMS Export"
    ])

    # ========================================================
    # TAB 1: Attendance Roster
    # ========================================================
    with att_tab1:
        f_col1, f_col2, f_col3 = st.columns([1.5, 1.2, 1.2])
        with f_col1:
            search_query = st.text_input("🔍 Search by Student Name, ID, or Track #", placeholder="e.g. Alice, STU-001, Track #2")
        with f_col2:
            status_filter = st.multiselect("Filter by Status", options=["Present", "Late", "Needs Review", "Absent"], default=["Present", "Late", "Needs Review", "Absent"])
        with f_col3:
            verif_filter = st.multiselect("Filter Verification", options=["face_matched", "anonymous", "manual_override"], default=["face_matched", "anonymous", "manual_override"])

        filtered = []
        for r in records:
            if r["status"] not in status_filter:
                continue
            if r.get("verification_type", "anonymous") not in verif_filter:
                continue
            if search_query:
                q = search_query.lower()
                sid_match = r.get("student_id") and q in r["student_id"].lower()
                name_match = r.get("student_name") and q in r["student_name"].lower()
                track_match = q in f"track #{r['track_id']}".lower()
                if not (sid_match or name_match or track_match):
                    continue
            filtered.append(r)

        c_v1, c_v2 = st.columns([3, 1])
        with c_v1:
            st.markdown(f"<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Verified Roster ({len(filtered)} / {len(records)} participants)</h4>", unsafe_allow_html=True)
        with c_v2:
            view_mode = st.radio("View Layout", ["📋 Table", "🪪 Cards"], horizontal=True, label_visibility="collapsed", key="att_view_mode")

        if not filtered:
            st.info("No records match your filter criteria.")
        else:
            df = pd.DataFrame(filtered)
            df["Participant"] = df.apply(lambda r: r["student_id"] if r["student_id"] else f"Anonymous Track #{r['track_id']}", axis=1)
            df["First Seen"] = df["first_seen_timestamp"].apply(format_seconds_to_mmss)
            df["Last Seen"] = df["last_seen_timestamp"].apply(format_seconds_to_mmss)
            df["Visible Duration"] = df["total_visible_seconds"].apply(format_seconds_to_mmss)
            df["Presence %"] = (df["presence_ratio"] * 100).round(1).astype(str) + "%"
            df["Match Conf"] = (df["confidence_avg"] * 100).round(1).astype(str) + "%"

            if view_mode == "📋 Table":
                display_cols = ["id", "Participant", "student_name", "status", "First Seen", "Last Seen", "Visible Duration", "Presence %", "verification_type", "Match Conf", "notes"]
                st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True)
            else:
                card_cols = st.columns(3)
                for idx, r in enumerate(filtered):
                    badge_class = "badge-present" if r["status"] == "Present" else ("badge-late" if r["status"] == "Late" else ("badge-absent" if r["status"] == "Absent" else "badge-review"))
                    icon = "✅" if r["status"] == "Present" else ("⏰" if r["status"] == "Late" else ("❌" if r["status"] == "Absent" else "🔍"))
                    part_label = r["student_id"] if r["student_id"] else f"Anonymous Track #{r['track_id']}"
                    s_name = r.get("student_name") or "Unenrolled / Consented Match"
                    pct = int(r["presence_ratio"] * 100)

                    with card_cols[idx % 3]:
                        st.markdown(
                            f"""
                            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08);
                                        border-radius: 10px; padding: 16px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-weight: 700; color: #f8fafc; font-size: 1.0rem;">{icon} {part_label}</span>
                                    <span class="{badge_class}">{r['status']}</span>
                                </div>
                                <div style="color: #94a3b8; font-size: 0.84rem; margin-bottom: 8px;">{s_name}</div>
                                <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;">Presence: <b>{pct}%</b> (Visible: {format_seconds_to_mmss(r['total_visible_seconds'])})</div>
                                <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; width: 100%; margin-bottom: 8px;">
                                    <div style="background: #38bdf8; height: 6px; border-radius: 4px; width: {min(100, pct)}%;"></div>
                                </div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #64748b;">
                                    <span>Seen: {format_seconds_to_mmss(r['first_seen_timestamp'])} - {format_seconds_to_mmss(r['last_seen_timestamp'])}</span>
                                    <span>Conf: {int(r['confidence_avg']*100)}%</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # ========================================================
    # TAB 2: Manual Overrides & Reassignment
    # ========================================================
    with att_tab2:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Instructor Status Corrections & Anonymous Reassignment</h4>", unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            st.markdown("<h5 style='color: #cbd5e1; font-size: 0.95rem; font-weight: 700;'>Manual Status Override</h5>", unsafe_allow_html=True)
            def format_rec_label(rid: int) -> str:
                rec = next((r for r in records if r["id"] == rid), None)
                if not rec:
                    return f"Record #{rid}"
                p_name = rec["student_id"] if rec["student_id"] else f"Track #{rec['track_id']}"
                return f"Record #{rid}: {p_name} (Current: {rec['status']})"

            with st.form("manual_override_form"):
                rec_to_edit = st.selectbox("Select Record to Modify:", options=[r["id"] for r in records], format_func=format_rec_label)
                new_status = st.selectbox("New Attendance Status", ["Present", "Late", "Needs Review", "Absent"])
                override_notes = st.text_input("Instructor Rationale", "Verified manually by lecturer")
                submit_ov = st.form_submit_button("Save Correction", type="primary")

                if submit_ov:
                    success = attendance_service.manual_override(rec_to_edit, new_status, override_notes)
                    if success:
                        st.success(f"Record #{rec_to_edit} updated to '{new_status}'.")
                        st.rerun()

        with c_act2:
            st.markdown("<h5 style='color: #cbd5e1; font-size: 0.95rem; font-weight: 700;'>Reassign Anonymous Track to Student</h5>", unsafe_allow_html=True)
            anon_records = [r for r in records if not r.get("student_id")]
            all_students = repo.list_students(active_only=True)

            if not anon_records:
                st.info("No anonymous tracks detected in this session.")
            elif not all_students:
                st.warning("No registered students found in directory.")
            else:
                with st.form("reassign_track_form"):
                    anon_to_reassign = st.selectbox(
                        "Select Anonymous Track:",
                        options=[r["id"] for r in anon_records],
                        format_func=lambda rid: f"Track #{next(r['track_id'] for r in anon_records if r['id'] == rid)} (Visible: {format_seconds_to_mmss(next(r['total_visible_seconds'] for r in anon_records if r['id'] == rid))})"
                    )
                    target_student = st.selectbox(
                        "Reassign To Registered Student:",
                        options=[s["student_id"] for s in all_students],
                        format_func=lambda sid: f"{sid} • {next(s['name'] for s in all_students if s['student_id'] == sid)}"
                    )
                    reassign_note = st.text_input("Reassignment Reason", "Instructor manually identified participant")
                    submit_reassign = st.form_submit_button("Confirm Reassignment", type="primary")

                    if submit_reassign:
                        ok = repo.reassign_attendance_record(anon_to_reassign, target_student, reassign_note)
                        if ok:
                            st.success(f"Track reassigned to student {target_student} successfully!")
                            st.rerun()

    # ========================================================
    # TAB 3: Quick Student Enrollment
    # ========================================================
    with att_tab3:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Fast In-Class Student Enrollment</h4>", unsafe_allow_html=True)
        st.caption("Enroll a student on the fly with reference photo capture and affirmative biometric consent.")

        with st.form("quick_enroll_form"):
            qc1, qc2 = st.columns(2)
            with qc1:
                q_sid = st.text_input("Student ID *", placeholder="e.g. STU-2026-006")
                q_name = st.text_input("Full Display Name *", placeholder="e.g. Jordan Lee")
            with qc2:
                q_email = st.text_input("Email Address", placeholder="e.g. jordan.lee@university.edu")
                q_courses = repo.list_courses()
                q_cmap = {c["id"]: f"{c['code']} - {c['name']}" for c in q_courses}
                q_cid = st.selectbox("Course", options=list(q_cmap.keys()), format_func=lambda cid: q_cmap[cid]) if q_courses else None

            q_consent = st.checkbox("Participant has provided affirmative consent for local facial recognition enrollment.")
            q_photo = st.file_uploader("Upload Clear Frontal Photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])

            if st.form_submit_button("Enroll Student Immediately", type="primary"):
                if not q_sid or not q_name:
                    st.error("Student ID and Full Name are required.")
                else:
                    img_b = q_photo.read() if q_photo else None
                    ok, msg = attendance_service.register_student(q_sid, q_name, q_email, q_cid, img_b, q_consent)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    # ========================================================
    # TAB 4: Official Reports & LMS Export
    # ========================================================
    with att_tab4:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Official Reports & Multi-Format Exports</h4>", unsafe_allow_html=True)
        st.caption("Export verified attendance records for academic records and Canvas / Moodle LMS gradebooks.")

        c_exp1, c_exp2, c_exp3 = st.columns(3)

        with c_exp1:
            csv_data = report_service.generate_attendance_csv(active_sid)
            st.download_button(
                label="📄 Export Attendance CSV",
                data=csv_data,
                file_name=f"AURORA_Attendance_Session_{active_sid}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with c_exp2:
            pdf_path = report_service.generate_session_pdf(active_sid)
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📑 Download Academic PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    use_container_width=True
                )

        with c_exp3:
            # Generate LMS JSON
            lms_export = {
                "system": "AURORA Multimodal Intelligence Agent",
                "session_id": active_sid,
                "session_name": session.get("session_name", "") if session else "",
                "course": session.get("course_name", "") if session else "",
                "date": session.get("date", "") if session else "",
                "duration_seconds": session.get("duration_seconds", 0) if session else 0,
                "attendance_rate_pct": rate,
                "total_records": len(records),
                "records": [
                    {
                        "student_id": r.get("student_id"),
                        "student_name": r.get("student_name"),
                        "status": r.get("status"),
                        "presence_ratio": r.get("presence_ratio"),
                        "total_visible_seconds": r.get("total_visible_seconds"),
                        "verification_type": r.get("verification_type")
                    }
                    for r in records
                ]
            }
            json_str = json.dumps(lms_export, indent=2)
            st.download_button(
                label="🌐 Export LMS JSON (Canvas/Moodle)",
                data=json_str,
                file_name=f"AURORA_LMS_Export_Session_{active_sid}.json",
                mime="application/json",
                use_container_width=True
            )
