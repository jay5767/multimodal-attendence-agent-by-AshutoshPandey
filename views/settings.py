"""Settings, Privacy policies, and System Configuration page in AURORA."""

import streamlit as st
import json
from config.settings import DEFAULT_CONFIG, DATABASE_PATH, MODELS_DIR
from database.repository import repo
from database.db import db_manager
from ui.theme import render_aurora_hero


def render_settings() -> None:
    render_aurora_hero(
        title="SETTINGS & PRIVACY GOVERNANCE",
        subtitle="Configure algorithmic thresholds, multimodal fusion weights, biometric policies, and inspect system health",
        badge="LOCAL DATA GOVERNANCE"
    )

    tab_att, tab_fusion, tab_privacy, tab_system = st.tabs([
        "⏱️ Attendance Rules",
        "⚖️ Fusion Weights",
        "🛡️ Privacy & GDPR",
        "💾 System Diagnostics"
    ])

    # ========================================================
    # TAB 1: Attendance Rules
    # ========================================================
    with tab_att:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Configurable Sustained-Presence Rules</h4>", unsafe_allow_html=True)
        st.caption("Thresholds required to mark a participant as 'Present' or 'Late' rather than 'Absent' or 'Needs Review'.")

        c1, c2 = st.columns(2)
        with c1:
            min_pres = st.slider(
                "Minimum Presence Ratio (% of Session)",
                min_value=10, max_value=90,
                value=int(DEFAULT_CONFIG.attendance.min_presence_ratio * 100),
                step=5
            )
            min_sec = st.number_input(
                "Minimum Continuous Visible Seconds",
                min_value=5.0, max_value=300.0,
                value=float(DEFAULT_CONFIG.attendance.min_visible_seconds),
                step=5.0
            )
        with c2:
            late_ratio = st.slider(
                "Late Arrival Cutoff (% of Session Elapsed)",
                min_value=5, max_value=50,
                value=int(DEFAULT_CONFIG.attendance.late_threshold_ratio * 100),
                step=5
            )
            face_sim = st.slider(
                "Face Match Similarity Threshold",
                min_value=0.30, max_value=0.90,
                value=float(DEFAULT_CONFIG.attendance.face_similarity_threshold),
                step=0.05
            )

        if st.button("💾 Save Attendance Thresholds", type="primary", use_container_width=True):
            DEFAULT_CONFIG.attendance.min_presence_ratio = min_pres / 100.0
            DEFAULT_CONFIG.attendance.min_visible_seconds = min_sec
            DEFAULT_CONFIG.attendance.late_threshold_ratio = late_ratio / 100.0
            DEFAULT_CONFIG.attendance.face_similarity_threshold = face_sim
            st.success("✅ Sustained-presence attendance thresholds saved for subsequent sessions!")

    # ========================================================
    # TAB 2: Multimodal Fusion Weights
    # ========================================================
    with tab_fusion:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Multimodal Composite Activity Weights</h4>", unsafe_allow_html=True)
        st.caption("Adjust the relative importance of visual movement vs classroom acoustic speech activity.")

        w_vis = st.slider(
            "Visual Movement Weight ($w_{vis}$)",
            min_value=0.0, max_value=1.0,
            value=float(DEFAULT_CONFIG.fusion.weight_visual_movement),
            step=0.05
        )
        w_aud = st.slider(
            "Speech Activity Weight ($w_{aud}$)",
            min_value=0.0, max_value=1.0,
            value=float(DEFAULT_CONFIG.fusion.weight_speech_activity),
            step=0.05
        )

        total_w = w_vis + w_aud
        st.info(f"Effective Combined Weight: **{total_w:.2f}** (Scores are normalized by sum of weights: $C_t = \\frac{{w_{{vis}} \\cdot V_t + w_{{aud}} \\cdot A_t}}{{w_{{vis}} + w_{{aud}}}}$)")

        if st.button("💾 Save Multimodal Weights", type="primary", use_container_width=True):
            DEFAULT_CONFIG.fusion.weight_visual_movement = w_vis
            DEFAULT_CONFIG.fusion.weight_speech_activity = w_aud
            st.success("✅ Multimodal fusion weights updated successfully!")

    # ========================================================
    # TAB 3: Privacy & GDPR
    # ========================================================
    with tab_privacy:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Biometric Governance & Data Rights</h4>", unsafe_allow_html=True)
        st.info("🔒 **Local-First Architecture:** This system operates strictly offline. No student media, raw face frames, or audio recordings are uploaded to cloud APIs.")

        st.markdown("<h5 style='color: #cbd5e1; font-size: 0.95rem; font-weight: 700;'>Media Retention Policy</h5>", unsafe_allow_html=True)
        ret_days = st.number_input(
            "Derived Analytics Retention Period (Days)",
            min_value=7, max_value=365,
            value=DEFAULT_CONFIG.privacy.retention_period_days
        )

        st.markdown("<h5 style='color: #cbd5e1; font-size: 0.95rem; font-weight: 700; margin-top: 14px;'>GDPR Article 17: Global Biometric Purge</h5>", unsafe_allow_html=True)
        st.warning("⚠️ **Right to Erasure:** Permanently erases all enrolled facial embedding vectors across the entire database, reverting student records to non-biometric status.")

        if st.button("🚨 Purge All Enrolled Biometric Signatures", use_container_width=True):
            with db_manager.get_connection() as conn:
                conn.execute("UPDATE students SET face_encoding_json = NULL, face_encoding = NULL, has_biometric_consent = 0")
                conn.commit()
            st.success("✅ All student facial embeddings have been permanently erased from the local database.")

    # ========================================================
    # TAB 4: System Diagnostics
    # ========================================================
    with tab_system:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Local Environment & Health Status</h4>", unsafe_allow_html=True)
        
        yolo_cached = (MODELS_DIR / "yolov8n.pt").exists()
        yolo_size = (MODELS_DIR / "yolov8n.pt").stat().st_size / (1024 * 1024) if yolo_cached else 0.0
        db_size_kb = round(DATABASE_PATH.stat().st_size / 1024, 2) if DATABASE_PATH.exists() else 0.0

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 18px; margin-bottom: 16px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Database Location</div>
                        <div style="font-size: 0.88rem; color: #f8fafc; word-break: break-all;"><code>{DATABASE_PATH}</code></div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Database Size</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">{db_size_kb} KB</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">YOLOv8 Weights</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #34d399;">{'Cached (' + str(round(yolo_size, 1)) + ' MB)' if yolo_cached else 'Fallback mode'}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Execution Mode</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">Air-Gapped Offline</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        summary = repo.get_dashboard_summary()
        st.json({
            "registered_students": summary["total_students"],
            "total_sessions": summary["total_sessions"],
            "offline_mode": DEFAULT_CONFIG.privacy.local_only_mode,
            "database_initialized": DATABASE_PATH.exists()
        })
