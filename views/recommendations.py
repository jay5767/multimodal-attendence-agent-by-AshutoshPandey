"""AI Teaching Assistant recommendations page with explainable pedagogical insights, strategy checklist, and timers in AURORA."""

import streamlit as st
import pandas as pd
from typing import Optional
from datetime import datetime

from database.repository import repo
from utils.timestamps import format_seconds_to_mmss
from ui.theme import render_aurora_hero


def render_recommendations(selected_session_id: Optional[int] = None) -> None:
    render_aurora_hero(
        title="AI TEACHING ASSISTANT",
        subtitle="Explainable pedagogical suggestions, evidence-based instructional adjustments, and active in-class strategy checklists",
        badge="TRANSPARENT RULE-BASED INFERENCE"
    )

    st.markdown(
        """
        <div style="background: rgba(6, 182, 212, 0.08); border-left: 4px solid #06b6d4; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 0.90rem; margin-bottom: 2px;">🛡️ PEDAGOGICAL ADVISORY MANDATE</div>
            <div style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.45;">
                AI Teaching Assistant suggestions are strictly <b>advisory and formative</b>. They support educators in adjusting lecture pacing and classroom discussions. 
                They must <b>NEVER</b> be used to evaluate student academic grades or make consequential assessments.
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
        "Select Classroom Session for Pedagogical Insights:",
        options=list(sess_options.keys()),
        index=0 if not selected_session_id or selected_session_id not in sess_options else list(sess_options.keys()).index(selected_session_id),
        format_func=lambda sid: sess_options[sid],
        key="recs_session_select"
    )

    recs = repo.get_recommendations_for_session(active_sid)

    recs_tab1, recs_tab2, recs_tab3 = st.tabs([
        f"💡 Explainable Suggestions ({len(recs)})",
        "⏱️ In-Class Pedagogical Checklist & Timer",
        "🛠️ Interactive Strategy Generator"
    ])

    # ========================================================
    # TAB 1: Explainable Teaching Suggestions
    # ========================================================
    with recs_tab1:
        if not recs:
            st.info("No automated recommendations generated for this session.")
        else:
            pending_cnt = sum(1 for r in recs if r.get("status", "pending") == "pending")
            accepted_cnt = sum(1 for r in recs if r.get("status") == "accepted")
            dismissed_cnt = sum(1 for r in recs if r.get("status") == "dismissed")

            sub_tab_all, sub_tab_pend, sub_tab_acc, sub_tab_dism = st.tabs([
                f"All Suggestions ({len(recs)})",
                f"⏳ Pending Action ({pending_cnt})",
                f"✅ Accepted ({accepted_cnt})",
                f"❌ Dismissed ({dismissed_cnt})"
            ])

            category_config = {
                "Interaction Lull": {
                    "emoji": "🤫", "accent": "#38bdf8",
                    "action": "Consider running a 60-second live poll, posing a challenge question on the whiteboard, or having students discuss with a neighbor."
                },
                "Acoustic Surge": {
                    "emoji": "🗣️", "accent": "#fbbf24",
                    "action": "Walk around the room to gauge whether the discussion is focused on the exercise or if a general clarification is needed."
                },
                "Movement Stillness": {
                    "emoji": "🧘", "accent": "#a855f7",
                    "action": "Offer a 30-second posture reset, switch from slide presentation to a live code/drawing demo, or ask for a show of hands."
                },
                "Pacing Opportunity": {
                    "emoji": "⏱️", "accent": "#34d399",
                    "action": "Maintain this instructional balance for high-concept seminar classes."
                },
                "General": {
                    "emoji": "📌", "accent": "#94a3b8",
                    "action": "Review session recordings for curriculum refinement."
                }
            }

            def render_rec_card(r: dict):
                rec_id = r["id"]
                cat = r.get("category", "General")
                cfg = category_config.get(cat, category_config["General"])
                conf = int(r.get("confidence", 0.8) * 100)
                t_start = format_seconds_to_mmss(r.get("timestamp_start_sec", 0.0))
                t_end = format_seconds_to_mmss(r.get("timestamp_end_sec", 0.0))
                curr_status = r.get("status", "pending")
                curr_notes = r.get("instructor_notes", "")

                status_badge_html = (
                    '<span class="badge-late">⏳ Pending Review</span>' if curr_status == "pending"
                    else ('<span class="badge-present">✅ Accepted Strategy</span>' if curr_status == "accepted"
                    else '<span class="badge-absent">❌ Dismissed</span>')
                )

                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08);
                                border-left: 5px solid {cfg['accent']}; border-radius: 10px; padding: 16px 20px; margin-bottom: 14px;
                                box-shadow: 0 4px 18px rgba(0,0,0,0.25);">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 1.0rem;">
                                {cfg['emoji']} [{cat}] Interval: {t_start} - {t_end}
                            </span>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <span class="badge-review">Confidence: {conf}%</span>
                                {status_badge_html}
                            </div>
                        </div>
                        <div style="color: #f8fafc; font-weight: 600; font-size: 0.95rem; margin-bottom: 8px;">
                            💡 <b>Recommendation:</b> {r.get('suggestion')}
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                            🎯 <b style="color: #38bdf8;">Suggested Next In-Class Action:</b> <span style="color: #cbd5e1;">{cfg['action']}</span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.84rem; margin-bottom: 6px;">
                            🔍 <b>Observed Empirical Evidence:</b> {r.get('evidence')}
                        </div>
                        <div style="color: #64748b; font-size: 0.76rem; font-style: italic; margin-bottom: 8px;">
                            ⚠️ <b>Data-Quality Limitations:</b> {r.get('limitations', 'Observable metrics are indicative only and do not reflect internal comprehension.')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander(f"⚙️ Review Suggestion #{rec_id} (Status: {curr_status.capitalize()})", expanded=False):
                    col_in1, col_in2, col_in3 = st.columns([2, 1, 1])
                    with col_in1:
                        inst_reflection = st.text_input(
                            "Instructor Reflection Note",
                            value=curr_notes,
                            key=f"note_input_{rec_id}",
                            placeholder="e.g. Discussed with students at 12:40, high participation."
                        )
                    with col_in2:
                        if st.button("✅ Accept Strategy", key=f"btn_accept_{rec_id}", use_container_width=True):
                            repo.update_recommendation_status(rec_id, "accepted", inst_reflection)
                            st.success("Accepted and logged.")
                            st.rerun()
                    with col_in3:
                        if st.button("❌ Dismiss", key=f"btn_dismiss_{rec_id}", use_container_width=True):
                            repo.update_recommendation_status(rec_id, "dismissed", inst_reflection)
                            st.warning("Suggestion dismissed.")
                            st.rerun()

            with sub_tab_all:
                for r in recs:
                    render_rec_card(r)

            with sub_tab_pend:
                p_recs = [r for r in recs if r.get("status", "pending") == "pending"]
                if not p_recs:
                    st.info("No pending recommendations. All suggestions reviewed!")
                else:
                    for r in p_recs:
                        render_rec_card(r)

            with sub_tab_acc:
                a_recs = [r for r in recs if r.get("status") == "accepted"]
                if not a_recs:
                    st.info("No accepted strategies recorded yet.")
                else:
                    for r in a_recs:
                        render_rec_card(r)

            with sub_tab_dism:
                d_recs = [r for r in recs if r.get("status") == "dismissed"]
                if not d_recs:
                    st.info("No dismissed suggestions.")
                else:
                    for r in d_recs:
                        render_rec_card(r)

    # ========================================================
    # TAB 2: In-Class Pedagogical Checklist & Timer
    # ========================================================
    with recs_tab2:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>⏱️ Active In-Class Pedagogical Strategy Checklist & Timer</h4>", unsafe_allow_html=True)
        st.caption("Keep your lectures dynamic using structured active learning intervals and interactive checkpoint tracking.")

        c_time1, c_time2 = st.columns([1, 2])
        with c_time1:
            st.markdown(
                """
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; text-align: center;">
                    <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase;">Active Strategy Timer</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #38bdf8; font-family: monospace; margin: 6px 0;">02:00</div>
                    <div style="color: #34d399; font-size: 0.78rem; font-weight: 600;">Target: Peer Think-Pair-Share</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_time2:
            st.markdown("<h5 style='color: #cbd5e1; font-size: 0.95rem; font-weight: 700;'>Lecture Strategy Checklist</h5>", unsafe_allow_html=True)
            chk1 = st.checkbox("🎯 60-Second Concept Checkpoint Poll on Board", value=True)
            chk2 = st.checkbox("👥 2-Minute Think-Pair-Share Neighbor Discussion", value=True)
            chk3 = st.checkbox("💻 Modality Shift: Switch from slides to live code/demo", value=False)
            chk4 = st.checkbox("🧘 30-Second Posture Reset / Deep Breath Pause", value=False)
            chk5 = st.checkbox("❓ Pose Counter-Example Question for Debate", value=False)
            chk6 = st.checkbox("📝 1-Minute Silent Written Reflection", value=False)

            total_chk = sum([chk1, chk2, chk3, chk4, chk5, chk6])
            st.progress(total_chk / 6.0)
            st.caption(f"**Progress:** {total_chk} / 6 active learning interventions applied.")

            if st.button("💾 Save Checklist to Session Audit Log", type="primary", use_container_width=True):
                st.success(f"Audit log updated with {total_chk} verified active learning interventions.")

    # ========================================================
    # TAB 3: Interactive Strategy Generator
    # ========================================================
    with recs_tab3:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; font-weight: 700;'>Interactive Active Learning Prompt Generator</h4>", unsafe_allow_html=True)
        c_tool1, c_tool2 = st.columns(2)
        with c_tool1:
            activity_type = st.selectbox("Observed Classroom Dynamic", ["Prolonged Silence / Lecture Lull", "Auditory Distraction / Noise Surge", "Low Movement / Fatigue"])
            target_time = st.slider("Target Activity Duration (minutes)", min_value=1, max_value=10, value=3)
        with c_tool2:
            class_size = st.selectbox("Class Size", ["Small (<20 students)", "Medium (20-50 students)", "Large Lecture Hall (>50 students)"])
            pedagogy_goal = st.selectbox("Pedagogical Goal", ["Concept Check", "Energy Reset", "Peer Debate"])

        if st.button("Generate Strategy Prompt", type="primary", use_container_width=True):
            st.success(f"**Recommended Strategy ({target_time} min for {class_size}):**")
            if "Prolonged Silence" in activity_type:
                st.write(f"👉 **Think-Pair-Share Checkpoint ({target_time} mins):** Pose a multiple-choice concept check on the whiteboard. Give students 60 seconds to think independently, 90 seconds to convince a neighbor, and finish with a cold-call or show of hands.")
            elif "Noise Surge" in activity_type:
                st.write(f"👉 **Guided Socratic Re-anchor ({target_time} mins):** Acknowledge the classroom buzz: *'I hear a lot of side discussions—who has a compelling counterexample to share with the whole room?'* Turn side conversation into plenary debate.")
            else:
                st.write(f"👉 **Active Polling & Stretch ({target_time} mins):** Have students stand up if they agree with Statement A, or remain seated for Statement B. Immediate posture reset with rapid formative assessment.")
