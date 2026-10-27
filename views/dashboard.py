"""Overview Dashboard page displaying real session KPIs, attendance stats, and activity trends in AURORA."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
from datetime import datetime

from database.repository import repo
from utils.timestamps import format_seconds_to_mmss
from ui.theme import render_aurora_hero, get_aurora_plotly_layout, render_empty_state
from config.settings import SAMPLES_DIR
from utils.sample_generator import generate_sample_classroom_video
from core.video_processor import VideoProcessingPipeline


def render_dashboard(selected_session_id: Optional[int] = None) -> None:
    render_aurora_hero(
        title="AURORA",
        subtitle="Multimodal Attendance & Observable Engagement Intelligence System",
        badge="LOCAL AI PROCESSING ACTIVE"
    )

    summary = repo.get_dashboard_summary(selected_session_id)
    session_data = summary["selected_session"]
    att = summary["attendance_summary"]

    # Top KPI Metrics Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(label="Registered Students", value=summary["total_students"])
    with kpi_col2:
        st.metric(label="Total Sessions", value=summary["total_sessions"])
    with kpi_col3:
        if session_data:
            pct = att["attendance_pct"]
            st.metric(label="Session Attendance Rate", value=f"{pct}%", delta=f"{att['present']} Present • {att['late']} Late")
        else:
            st.metric(label="Session Attendance Rate", value="N/A")
    with kpi_col4:
        if session_data:
            avg_act = summary["avg_activity_index"]
            st.metric(label="Avg Observable Activity", value=f"{avg_act} / 1.0", delta="Multi-window avg")
        else:
            st.metric(label="Avg Observable Activity", value="N/A")

    st.markdown("<div style='margin: 18px 0; height: 1px; background: rgba(255,255,255,0.07);'></div>", unsafe_allow_html=True)

    # Empty State Onboarding if no sessions exist
    if not session_data:
        render_empty_state(
            title="No Classroom Sessions Ingested Yet",
            message="Welcome to AURORA! Ingest recorded classroom footage or generate an instant synthetic benchmark session to explore automated face attendance, observable engagement metrics, and explainable AI teaching insights."
        )
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("⚡ Generate & Process Synthetic Benchmark (15s)", type="primary", use_container_width=True):
                with st.spinner("Synthesizing benchmark classroom video, audio, and participant avatars..."):
                    sample_file = SAMPLES_DIR / "classroom_benchmark_lecture.mp4"
                    generate_sample_classroom_video(sample_file, duration_seconds=15, fps=15)
                    c_list = repo.list_courses()
                    c_id = c_list[0]["id"] if c_list else repo.create_course("CS302", "Multimodal Classroom AI", "Prof. Vance")
                    sess_id = repo.create_session(
                        session_name="Lecture 1: Foundations of Multimodal Sensing",
                        course_id=c_id,
                        instructor="Prof. Elena Vance",
                        date=datetime.now().strftime("%Y-%m-%d"),
                        video_path=str(sample_file)
                    )
                    pipeline = VideoProcessingPipeline()
                    pipeline.process_session(sess_id, sample_file)
                    st.success(f"Benchmark session #{sess_id} initialized!")
                    st.rerun()
        with col_act2:
            st.info("💡 You can also navigate to **Video & Live Analysis** to upload your own MP4, AVI, or MOV classroom lecture.")
        return

    # Selected Session Overview Card
    dur_str = format_seconds_to_mmss(session_data.get('duration_seconds', 0.0))
    fps_val = session_data.get('fps', 0.0)
    res_val = session_data.get('resolution') or '1280x720'
    c_name = session_data.get('course_name') or 'General Course'
    inst_name = session_data.get('instructor') or 'Not Specified'
    dt_str = session_data.get('date', 'N/A')

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.75) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-top: 3px solid #06b6d4;
                    border-radius: 12px; padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">ACTIVE CLASSROOM SESSION</span>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 4px 0 10px 0;">{session_data.get('session_name', 'Unnamed Session')}</div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="badge-present">Status: {session_data.get('status', 'completed').capitalize()}</span>
                    <span class="badge-review">ID: #{session_data.get('id')}</span>
                </div>
            </div>
            <div style="display: flex; gap: 24px; font-size: 0.88rem; color: #cbd5e1; flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 12px;">
                <span>📚 <b>Course:</b> {c_name}</span>
                <span>👨‍🏫 <b>Instructor:</b> {inst_name}</span>
                <span>📅 <b>Date:</b> {dt_str}</span>
                <span>⏱️ <b>Duration:</b> {dur_str}</span>
                <span>🎞️ <b>FPS:</b> {fps_val:.1f}</span>
                <span>🖥️ <b>Resolution:</b> {res_val}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Attendance Breakdown & Activity Gauge
    c_chart1, c_chart2 = st.columns([1, 1])

    with c_chart1:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;'>Classroom Attendance Distribution</h4>", unsafe_allow_html=True)
        if att["total"] > 0:
            raw_slices = [
                ("Present", att["present"], "#10b981"),
                ("Late", att["late"], "#f59e0b"),
                ("Absent", att["absent"], "#ef4444"),
                ("Needs Review", att["needs_review"], "#6366f1")
            ]
            non_zero_slices = [s for s in raw_slices if s[1] > 0]
            if not non_zero_slices:
                non_zero_slices = raw_slices

            labels, values, colors_list = zip(*non_zero_slices)

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors_list, line=dict(color="#090d16", width=2)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                textposition="inside",
                textfont=dict(color="#ffffff", size=11, family="'Plus Jakarta Sans', sans-serif")
            )])
            layout_donut = get_aurora_plotly_layout(height=260)
            layout_donut["showlegend"] = True
            layout_donut["legend"] = dict(
                orientation="h",
                yanchor="top",
                y=-0.02,
                xanchor="center",
                x=0.5,
                font=dict(color="#cbd5e1", size=10)
            )
            fig_donut.add_annotation(
                text=f"<b style='font-size:18px; color:#f8fafc;'>{att['attendance_pct']}%</b><br><span style='font-size:10px; color:#94a3b8;'>ATTENDANCE</span>",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(family="'Plus Jakarta Sans', sans-serif")
            )
            fig_donut.update_layout(**layout_donut)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No attendance records computed for this session.")

    with c_chart2:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;'>Observable Activity Level</h4>", unsafe_allow_html=True)
        avg_score = summary["avg_activity_index"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Activity Index", 'font': {'size': 13, 'color': '#94a3b8'}},
            number={'font': {'color': '#38bdf8', 'size': 34, 'family': "'JetBrains Mono', monospace"}, 'valueformat': '.2f'},
            gauge={
                'axis': {
                    'range': [0, 1.0],
                    'tickmode': 'array',
                    'tickvals': [0.0, 0.5, 1.0],
                    'ticktext': ['0.0', '0.5', '1.0'],
                    'tickwidth': 1,
                    'tickcolor': "#64748b",
                    'tickfont': {'color': '#94a3b8', 'size': 11}
                },
                'bar': {'color': "#06b6d4", 'thickness': 0.35},
                'bgcolor': "rgba(15, 23, 42, 0.6)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 0.3], 'color': 'rgba(100, 116, 139, 0.25)'},
                    {'range': [0.3, 0.7], 'color': 'rgba(6, 182, 212, 0.20)'},
                    {'range': [0.7, 1.0], 'color': 'rgba(16, 185, 129, 0.25)'}
                ],
            }
        ))
        layout_gauge = get_aurora_plotly_layout(height=260)
        layout_gauge["margin"] = dict(t=35, b=10, l=35, r=35)
        fig_gauge.update_layout(**layout_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Activity Timeline Chart
    st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700; margin-top: 14px; margin-bottom: 8px;'>Classroom Activity Timeline Across Discrete Time Windows</h4>", unsafe_allow_html=True)
    fused_metrics = repo.get_fused_metrics_for_session(session_data["id"])

    if fused_metrics:
        df_fused = pd.DataFrame(fused_metrics)
        df_fused["time_label"] = df_fused["window_start_sec"].apply(format_seconds_to_mmss)

        fig_timeline = go.Figure()

        # Composite score filled line
        fig_timeline.add_trace(go.Scatter(
            x=df_fused["time_label"],
            y=df_fused["composite_activity_score"],
            mode="lines+markers",
            name="Composite Activity Score",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            marker=dict(size=6, color="#0284c7")
        ))

        # Visual index
        fig_timeline.add_trace(go.Scatter(
            x=df_fused["time_label"],
            y=df_fused["visual_activity_index"],
            mode="lines",
            name="Visual Movement Index",
            line=dict(color="#34d399", width=2, dash="dot")
        ))

        # Speech ratio
        fig_timeline.add_trace(go.Scatter(
            x=df_fused["time_label"],
            y=df_fused["speech_activity_index"],
            mode="lines",
            name="Speech Activity Ratio",
            line=dict(color="#fbbf24", width=2, dash="dash")
        ))

        layout_timeline = get_aurora_plotly_layout(height=320)
        layout_timeline["xaxis_title"] = "Session Timeline (MM:SS)"
        layout_timeline["yaxis_title"] = "Score (0.0 to 1.0)"
        layout_timeline["yaxis"] = dict(range=[0, 1.05], gridcolor="rgba(255, 255, 255, 0.05)")
        layout_timeline["hovermode"] = "x unified"
        fig_timeline.update_layout(**layout_timeline)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No windowed metrics computed for this session yet.")

    # AI Teaching Recommendations Highlights
    st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700; margin-top: 14px; margin-bottom: 8px;'>💡 AI Teaching Assistant Insights</h4>", unsafe_allow_html=True)
    recs = repo.get_recommendations_for_session(session_data["id"])

    if recs:
        for r in recs[:3]:  # Show top 3 in dashboard
            conf = int(r.get("confidence", 0.8) * 100)
            t_span = f"{format_seconds_to_mmss(r['timestamp_start_sec'])} - {format_seconds_to_mmss(r['timestamp_end_sec'])}"
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid #38bdf8;
                            border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">📌 [{r['category']}] Interval: {t_span}</span>
                        <span class="badge-review">Confidence: {conf}%</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.92rem; font-weight: 500; margin-bottom: 6px;">
                        💡 <b>Recommendation:</b> {r.get('suggestion')}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.82rem;">
                        🔍 <b>Observed Evidence:</b> {r.get('evidence')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.caption("Navigate to **AI Teaching Assistant** in the sidebar to review, accept, or dismiss pedagogical action plans.")
    else:
        st.info("No specific pedagogical intervention triggers detected for this session.")
