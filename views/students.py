"""Student Management page for enrollment, consented biometric registration, and attendance histories in AURORA."""

import streamlit as st
import pandas as pd
from typing import Optional

from database.repository import repo
from services.attendance_service import attendance_service
from utils.timestamps import format_seconds_to_mmss
from ui.theme import render_aurora_hero


def render_students() -> None:
    render_aurora_hero(
        title="STUDENT MANAGEMENT",
        subtitle="Manage student records, capture consented biometric templates, and audit individual attendance histories",
        badge="PRIVACY-FIRST BIOMETRICS"
    )

    tab_list, tab_register, tab_history = st.tabs([
        "📋 Student Directory",
        "➕ Register & Enroll Student",
        "📈 Attendance History"
    ])

    # ========================================================
    # TAB 1: Student Directory
    # ========================================================
    with tab_list:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Enrolled Student Roster</h4>", unsafe_allow_html=True)
        students = repo.list_students(active_only=False)
        if not students:
            st.info("No registered students found. Use the **'Register & Enroll Student'** tab to add students.")
        else:
            s_filter = st.text_input("🔍 Filter Students by Name or ID", placeholder="e.g. Alice, STU-2026-001")
            
            df_students = pd.DataFrame(students)
            if s_filter:
                q = s_filter.lower()
                df_students = df_students[
                    df_students["name"].str.lower().str.contains(q, na=False) |
                    df_students["student_id"].str.lower().str.contains(q, na=False)
                ]

            df_students["Biometrics"] = df_students["has_biometric_consent"].apply(lambda x: "✅ Consented" if x else "❌ None")
            df_students["Status"] = df_students["is_active"].apply(lambda x: "Active" if x else "Inactive")

            cols_to_show = ["student_id", "name", "email", "course_name", "Biometrics", "Status", "created_at"]
            st.dataframe(df_students[[c for c in cols_to_show if c in df_students.columns]], use_container_width=True)

            st.markdown("<h5 style='color: #f8fafc; font-size: 0.95rem; font-weight: 700; margin-top: 16px;'>Student Governance Actions</h5>", unsafe_allow_html=True)
            selected_sid = st.selectbox(
                "Select Student to Manage:",
                options=[s["student_id"] for s in students],
                format_func=lambda sid: f"{sid} • {next(s['name'] for s in students if s['student_id'] == sid)}",
                key="manage_student_select"
            )

            if selected_sid:
                student_info = next(s for s in students if s["student_id"] == selected_sid)
                col_act1, col_act2, col_act3 = st.columns(3)

                with col_act1:
                    if st.button("🗑️ Purge Biometric Data", help="Permanently erase enrolled facial embedding vectors (GDPR Right-to-Erasure)", use_container_width=True):
                        success = attendance_service.purge_student_biometrics(selected_sid)
                        if success:
                            st.success(f"Biometric signatures for {selected_sid} erased from local database.")
                            st.rerun()

                with col_act2:
                    is_active = student_info.get("is_active", 1) == 1
                    toggle_label = "Deactivate Student" if is_active else "Reactivate Student"
                    if st.button(toggle_label, use_container_width=True):
                        repo.update_student(selected_sid, student_info["name"], student_info.get("email"), not is_active)
                        st.success(f"Status updated to {'Active' if not is_active else 'Inactive'}.")
                        st.rerun()

                with col_act3:
                    if st.button("❌ Delete Student Permanently", type="primary", use_container_width=True):
                        repo.delete_student(selected_sid)
                        st.warning(f"Student {selected_sid} permanently deleted.")
                        st.rerun()

    # ========================================================
    # TAB 2: Register New Student
    # ========================================================
    with tab_register:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Register & Enroll New Student</h4>", unsafe_allow_html=True)
        with st.form("register_student_form", clear_on_submit=False):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                student_id = st.text_input("Student ID *", placeholder="e.g. STU-2026-005")
                name = st.text_input("Full Display Name *", placeholder="e.g. Maya Lin")
            with r_col2:
                email = st.text_input("University Email Address", placeholder="e.g. maya.lin@university.edu")
                courses = repo.list_courses()
                course_options = {c["id"]: f"{c['code']} - {c['name']}" for c in courses}
                course_id = st.selectbox("Assign Course", options=list(course_options.keys()), format_func=lambda cid: course_options[cid]) if courses else None

            st.markdown("<div style='margin: 14px 0; height: 1px; background: rgba(255,255,255,0.07);'></div>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #38bdf8; font-size: 0.95rem; font-weight: 700;'>Optional Facial Recognition Enrollment</h5>", unsafe_allow_html=True)
            st.caption("Facial embedding vectors are computed locally and stored only in the encrypted local database. Never sent to cloud servers.")
            
            consent_granted = st.checkbox("Participant has provided affirmative consent for local facial recognition enrollment.")

            photo_mode = st.radio("Reference Photo Mode:", ["📁 Upload Photo File", "📸 Capture with Webcam"], horizontal=True, key="photo_input_mode")
            img_bytes = None

            if photo_mode == "📁 Upload Photo File":
                uploaded_photo = st.file_uploader("Upload Clear Frontal Face Photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_photo:
                    img_bytes = uploaded_photo.read()
            else:
                cam_snap = st.camera_input("Take Enrollment Photo")
                if cam_snap:
                    img_bytes = cam_snap.getvalue()

            submitted = st.form_submit_button("Register & Enroll Student", type="primary", use_container_width=True)

            if submitted:
                if not student_id or not name:
                    st.error("Please fill in Student ID and Full Name.")
                else:
                    success, msg = attendance_service.register_student(
                        student_id=student_id,
                        name=name,
                        email=email,
                        course_id=course_id,
                        reference_image_bytes=img_bytes,
                        has_consent=consent_granted
                    )
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

    # ========================================================
    # TAB 3: Attendance History
    # ========================================================
    with tab_history:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Individual Student Attendance History</h4>", unsafe_allow_html=True)
        all_s = repo.list_students(active_only=False)
        if not all_s:
            st.info("No students registered.")
        else:
            h_sid = st.selectbox(
                "Select Student to Audit:",
                options=[s["student_id"] for s in all_s],
                format_func=lambda sid: f"{sid} • {next(s['name'] for s in all_s if s['student_id'] == sid)}",
                key="history_select"
            )
            if h_sid:
                stats = attendance_service.get_student_statistics(h_sid)
                h_col1, h_col2, h_col3, h_col4 = st.columns(4)
                with h_col1:
                    st.metric("Total Sessions Attended", stats["total_sessions"])
                with h_col2:
                    st.metric("Cumulative Attendance Rate", f"{stats['attendance_rate']}%")
                with h_col3:
                    st.metric("Present / Late", f"{stats['present_count']} / {stats['late_count']}")
                with h_col4:
                    st.metric("Absent / Incomplete", stats["absent_count"])

                if stats["records"]:
                    df_h = pd.DataFrame(stats["records"])
                    df_h["Visible Time"] = df_h["total_visible_seconds"].apply(format_seconds_to_mmss)
                    df_h["Presence %"] = (df_h["presence_ratio"] * 100).round(1).astype(str) + "%"
                    cols = ["session_name", "date", "status", "Visible Time", "Presence %", "verification_type", "notes"]
                    st.dataframe(df_h[[c for c in cols if c in df_h.columns]], use_container_width=True)
                else:
                    st.info("No recorded session attendance for this student yet.")
