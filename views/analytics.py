"""Engagement & Multimodal Fusion Analytics page with interactive time-window inspector and classroom seating heatmap in AURORA."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional

from database.repository import repo
from services.report_service import report_service
from utils.timestamps import format_seconds_to_mmss
from ui.theme import render_aurora_hero, get_aurora_plotly_layout


def render_analytics(selected_session_id: Optional[int] = None) -> None:
    render_aurora_hero(
        title="ENGAGEMENT & MULTIMODAL ANALYTICS",
        subtitle="Synchronized visual orientation, motion dynamics, acoustic speech rhythms, and classroom seating heatmaps",
        badge="SCIENTIFIC OBSERVABLE INDICATORS"
    )

    # Scientific Caution Alert
    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;">
            <div style="color: #fbbf24; font-weight: 700; font-size: 0.90rem; margin-bottom: 2px;">🔬 SCIENTIFIC NOTICE ON OBSERVABLE CLASSROOM SIGNALS</div>
            <div style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.45;">
                All measurements reported here represent <b>observable physical indicators</b> (head orientation, spatial pixel motion, acoustic sound energy). 
                They <b>DO NOT</b> claim to measure internal comprehension or cognitive states. Quiet stillness often indicates deep concentration during lectures.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    sessions = repo.list_sessions()
    if not sessions:
        st.info("No recorded sessions found. Go to **Video Processing** to ingest a lecture.")
        return

    sess_options = {s["id"]: f"Session #{s['id']} • {s['session_name']} ({s['date']})" for s in sessions}
    active_sid = st.selectbox(
        "Select Classroom Session for Multimodal Analytics:",
        options=list(sess_options.keys()),
        index=0 if not selected_session_id or selected_session_id not in sess_options else list(sess_options.keys()).index(selected_session_id),
        format_func=lambda sid: sess_options[sid],
        key="analytics_session_select"
    )

    fused_metrics = repo.get_fused_metrics_for_session(active_sid)
    visual_measurements = repo.get_visual_measurements_for_session(active_sid)
    audio_measurements = repo.get_audio_measurements_for_session(active_sid)
    attendance_records = repo.get_attendance_for_session(active_sid)

    if not fused_metrics:
        st.warning("No analytics metrics computed for this session yet.")
        return

    df_fused = pd.DataFrame(fused_metrics)
    df_fused["Time Window"] = df_fused["window_start_sec"].apply(format_seconds_to_mmss)

    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "📈 Synchronized Trends & Timeline",
        "🔍 Window Interval Inspector",
        "🪑 Classroom Seating Grid & Heatmap",
        "📊 Signal Dynamics & Quality"
    ])

    # ========================================================
    # TAB 1: Synchronized Trends & Timeline
    # ========================================================
    with analytics_tab1:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Continuous Multimodal Activity Trends</h4>", unsafe_allow_html=True)
        fig_fused = go.Figure()

        # Composite Activity Score
        fig_fused.add_trace(go.Scatter(
            x=df_fused["Time Window"],
            y=df_fused["composite_activity_score"],
            name="Composite Activity Score",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            mode="lines+markers",
            marker=dict(size=7, color="#0284c7")
        ))

        # Visual Movement Index
        fig_fused.add_trace(go.Scatter(
            x=df_fused["Time Window"],
            y=df_fused["visual_activity_index"],
            name="Visual Movement Index",
            line=dict(color="#34d399", width=2, dash="dash"),
            mode="lines"
        ))

        # Speech Activity Ratio
        fig_fused.add_trace(go.Scatter(
            x=df_fused["Time Window"],
            y=df_fused["speech_activity_index"],
            name="Speech Activity Ratio",
            line=dict(color="#fbbf24", width=2, dash="dot"),
            mode="lines"
        ))

        # Moving Average Trendline
        if len(df_fused) >= 3:
            rolling_score = df_fused["composite_activity_score"].rolling(window=3, min_periods=1).mean()
            fig_fused.add_trace(go.Scatter(
                x=df_fused["Time Window"],
                y=rolling_score,
                name="Smoothed Trendline (3-Win Moving Avg)",
                line=dict(color="#a855f7", width=1.8),
                mode="lines"
            ))

        layout_fused = get_aurora_plotly_layout(height=360, title="Continuous Activity Metrics Across Temporal Windows")
        layout_fused["xaxis_title"] = "Session Timeline (MM:SS)"
        layout_fused["yaxis_title"] = "Normalized Activity Score (0.0 to 1.0)"
        layout_fused["yaxis"] = dict(range=[0, 1.05], gridcolor="rgba(255, 255, 255, 0.05)")
        layout_fused["hovermode"] = "x unified"
        fig_fused.update_layout(**layout_fused)
        st.plotly_chart(fig_fused, use_container_width=True)

    # ========================================================
    # TAB 2: Interactive Window Interval Inspector
    # ========================================================
    with analytics_tab2:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Granular Interval Scrutiny & Landmark Inspection</h4>", unsafe_allow_html=True)
        st.caption("Select any discrete continuous window to dissect detected participant head orientation angles, velocities, and acoustic levels.")

        window_labels = [
            f"Window #{idx+1}: {format_seconds_to_mmss(w['window_start_sec'])} - {format_seconds_to_mmss(w['window_end_sec'])}"
            for idx, w in enumerate(fused_metrics)
        ]
        selected_win_idx = st.selectbox(
            "Select Window Interval to Inspect:",
            options=list(range(len(fused_metrics))),
            format_func=lambda i: window_labels[i],
            key="selected_interval_inspector"
        )

        selected_win = fused_metrics[selected_win_idx]
        w_start = selected_win["window_start_sec"]
        w_end = selected_win["window_end_sec"]

        win_visuals = repo.get_visual_measurements_for_window(active_sid, w_start, w_end)

        # Inspector KPI Box
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 18px 22px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">
                        Window #{selected_win_idx+1}: {format_seconds_to_mmss(w_start)} - {format_seconds_to_mmss(w_end)} ({w_end - w_start:.1f}s span)
                    </span>
                    <div style="display: flex; gap: 8px;">
                        <span class="badge-review">Signal: {selected_win.get('signal_availability', 'both').upper()}</span>
                        <span class="badge-present">Quality: {int(selected_win.get('data_quality_index', 1.0)*100)}%</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;">
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Composite Score</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">{selected_win['composite_activity_score']:.2f}</div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Visual Motion</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #34d399;">{selected_win['visual_activity_index']:.2f}</div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Speech Ratio</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #fbbf24;">{selected_win['speech_activity_index']:.2f}</div>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Active Participants</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{selected_win['active_participants_count']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if win_visuals:
            df_win_vis = pd.DataFrame(win_visuals)
            df_win_vis["Time"] = df_win_vis["timestamp_sec"].apply(format_seconds_to_mmss)
            df_win_vis["Track"] = df_win_vis["track_id"].apply(lambda t: f"Track #{t}")
            df_win_vis["Yaw (°)"] = df_win_vis["head_yaw"].apply(lambda y: f"{y:.1f}°" if y is not None else "N/A")
            df_win_vis["Pitch (°)"] = df_win_vis["head_pitch"].apply(lambda p: f"{p:.1f}°" if p is not None else "N/A")
            df_win_vis["Speed"] = df_win_vis["head_movement_speed"].apply(lambda s: f"{s:.3f}" if s is not None else "0.000")
            show_cols = ["Time", "Track", "observable_state", "Yaw (°)", "Pitch (°)", "Speed", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
            st.dataframe(df_win_vis[[c for c in show_cols if c in df_win_vis.columns]], use_container_width=True)
        else:
            st.info("No individual visual detections recorded in this time window.")

    # ========================================================
    # TAB 3: Classroom Seating Grid & Heatmap
    # ========================================================
    with analytics_tab3:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Classroom Spatial Seating Layout & Engagement Map</h4>", unsafe_allow_html=True)
        st.caption("Visual reconstruction of classroom participant seating positions and physical engagement indicators.")

        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px dashed rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 16px;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.08em;">INSTRUCTIONAL WHITEBOARD / PODIUM (FRONT OF ROOM)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Build virtual desks from attendance records
        if attendance_records:
            desk_cols = st.columns(3)
            for idx, r in enumerate(attendance_records):
                p_label = r.get("student_name") or (f"Student {r['student_id']}" if r.get("student_id") else f"Anonymous Track #{r['track_id']}")
                pct = int(r.get("presence_ratio", 0.0) * 100)
                status = r.get("status", "Present")

                desk_state = "High Focus" if status == "Present" else ("Late Arrival" if status == "Late" else ("Needs Review" if status == "Needs Review" else "Absent"))
                desk_color = "#10b981" if status == "Present" else ("#f59e0b" if status == "Late" else ("#6366f1" if status == "Needs Review" else "#ef4444"))

                with desk_cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid {desk_color};
                                    border-radius: 10px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">Desk #{idx+1}</span>
                                <span style="color: {desk_color}; font-size: 0.75rem; font-weight: 700;">● {desk_state}</span>
                            </div>
                            <div style="color: #cbd5e1; font-size: 0.88rem; font-weight: 600; margin-bottom: 6px;">{p_label}</div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin-bottom: 4px;">
                                <span>Presence: <b>{pct}%</b></span>
                                <span>Visible: {format_seconds_to_mmss(r.get('total_visible_seconds', 0.0))}</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 5px; width: 100%;">
                                <div style="background: {desk_color}; height: 5px; border-radius: 4px; width: {min(100, pct)}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No participants tracked in this session.")

    # ========================================================
    # TAB 4: Signal Dynamics & Quality
    # ========================================================
    with analytics_tab4:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Observed Distributions & Signal Dynamics</h4>", unsafe_allow_html=True)
        c_vis, c_aud = st.columns(2)

        with c_vis:
            st.markdown("<h5 style='color: #cbd5e1; font-size: 0.92rem; font-weight: 700;'>Observed Head Orientation Distribution</h5>", unsafe_allow_html=True)
            if visual_measurements:
                df_vis_all = pd.DataFrame(visual_measurements)
                state_counts = df_vis_all["observable_state"].value_counts().reset_index()
                state_counts.columns = ["State", "Count"]

                fig_states = px.bar(
                    state_counts,
                    x="State",
                    y="Count",
                    color="State",
                    color_discrete_map={
                        "forward_facing": "#10b981",
                        "turned_away": "#f59e0b",
                        "looking_down": "#38bdf8",
                        "looking_up": "#8b5cf6",
                        "unavailable": "#64748b"
                    }
                )
                layout_states = get_aurora_plotly_layout(height=240)
                layout_states["showlegend"] = False
                fig_states.update_layout(**layout_states)
                st.plotly_chart(fig_states, use_container_width=True)
            else:
                st.info("No visual orientation measurements.")

        with c_aud:
            st.markdown("<h5 style='color: #cbd5e1; font-size: 0.92rem; font-weight: 700;'>Acoustic Sound Energy vs. Speech Activity</h5>", unsafe_allow_html=True)
            if audio_measurements:
                df_aud = pd.DataFrame(audio_measurements)
                df_aud["Time"] = df_aud["start_sec"].apply(format_seconds_to_mmss)

                fig_aud = go.Figure()
                fig_aud.add_trace(go.Bar(
                    x=df_aud["Time"],
                    y=df_aud["speech_ratio"],
                    name="Speech Ratio",
                    marker_color="#fbbf24"
                ))
                fig_aud.add_trace(go.Scatter(
                    x=df_aud["Time"],
                    y=df_aud["energy_level"],
                    name="RMS Energy",
                    yaxis="y2",
                    line=dict(color="#f43f5e", width=2)
                ))
                layout_aud = get_aurora_plotly_layout(height=240)
                layout_aud["yaxis"] = dict(title="Speech Ratio (0-1)", range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)")
                layout_aud["yaxis2"] = dict(title="RMS Energy", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)")
                fig_aud.update_layout(**layout_aud)
                st.plotly_chart(fig_aud, use_container_width=True)
            else:
                st.info("No audio measurements recorded.")

        # Export section
        st.markdown("<div style='margin: 14px 0; height: 1px; background: rgba(255,255,255,0.07);'></div>", unsafe_allow_html=True)
        c_q1, c_q2 = st.columns(2)
        with c_q1:
            avg_quality = round(df_fused["data_quality_index"].mean() * 100, 1)
            st.metric(label="Overall Signal Quality Index", value=f"{avg_quality}%", delta="Based on face visibility and audio track presence")
        with c_q2:
            csv_metrics = report_service.generate_fused_metrics_csv(active_sid)
            st.download_button(
                label="📊 Download Multimodal Metrics CSV",
                data=csv_metrics,
                file_name=f"AURORA_Multimodal_Analytics_Session_{active_sid}.csv",
                mime="text/csv",
                use_container_width=True
            )
