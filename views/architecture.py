"""System Architecture and Multimodal Workflow Interactive Visualizer in AURORA."""

import streamlit as st
from config.settings import DEFAULT_CONFIG, DATABASE_PATH, MODELS_DIR
from ui.theme import render_aurora_hero


def render_architecture() -> None:
    render_aurora_hero(
        title="SYSTEM ARCHITECTURE & WORKFLOW",
        subtitle="Interactive schematic of the offline multimodal pipeline, data transformations, and mathematical formulations",
        badge="TRANSPARENT SYSTEM DESIGN"
    )

    # Component Status Health Badges
    st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>1. Subsystem Health & Offline Model Cache</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    yolo_file = MODELS_DIR / "yolov8n.pt"
    yolo_status = "🟢 Cached Locally (6.5 MB)" if yolo_file.exists() else "🟡 Online fallback"

    with c1:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #38bdf8; padding: 14px; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">OBJECT DETECTOR</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">YOLOv8n + HOG</div>
                <div style="font-size: 0.76rem; color: #34d399; font-weight: 600;">{yolo_status}</div>
            </div>
            """, unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #34d399; padding: 14px; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">SPATIAL TRACKER</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">Spatial IoU + Trajectory</div>
                <div style="font-size: 0.76rem; color: #34d399; font-weight: 600;">🟢 Native Python (Real-Time)</div>
            </div>
            """, unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #fbbf24; padding: 14px; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">AUDIO VAD ENGINE</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">16kHz RMS + VAD</div>
                <div style="font-size: 0.76rem; color: #34d399; font-weight: 600;">🟢 Local PCM WAV Analyzer</div>
            </div>
            """, unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #8b5cf6; padding: 14px; border-radius: 10px;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">PERSISTENCE</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">SQLite WAL Mode</div>
                <div style="font-size: 0.76rem; color: #34d399; font-weight: 600;">🟢 Foreign Keys & Indexed</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<div style='margin: 20px 0; height: 1px; background: rgba(255,255,255,0.07);'></div>", unsafe_allow_html=True)

    # Workflow Pipeline Architecture
    st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>2. Pipeline Flow & Mathematical Modeling</h4>", unsafe_allow_html=True)

    tab_flow, tab_fusion, tab_presence, tab_privacy = st.tabs([
        "🔄 End-to-End Multimodal Dataflow",
        "⚖️ Composite Activity Formulation",
        "⏱️ Sustained-Presence State Logic",
        "🛡️ Privacy & Compliance Architecture"
    ])

    with tab_flow:
        st.markdown(
            """
            ```mermaid
            flowchart TD
                subgraph InputModality ["Input Acquisition Mode"]
                    A1["Recorded Video (MP4 / AVI / MOV)"]
                    A2["Live Classroom Webcam Stream"]
                    A3["Classroom Audio Track (16kHz WAV / Mic)"]
                end

                subgraph VisualStream ["Visual Signal Stream"]
                    B1["Frame Sampling (1-5 FPS)"]
                    B2["Person Detection (YOLOv8 + HOG Fallback)"]
                    B3["Spatial IoU Multi-Person Tracking"]
                    B4["Consented Facial Identity Matching"]
                    B5["Head Orientation (Yaw / Pitch / Roll)"]
                    B6["Inactivity & Movement Dynamics"]
                end

                subgraph AudioStream ["Acoustic Signal Stream"]
                    C1["PCM 16-bit Audio Extraction (ffmpeg)"]
                    C2["Sub-frame RMS Energy Calculation"]
                    C3["Voice Activity Detection (VAD)"]
                    C4["Silence & Discussion Intervals"]
                end

                subgraph MultimodalFusion ["Multimodal Intelligence Core"]
                    D1["Temporal Synchronization (30s Windows)"]
                    D2["Composite Activity Index Calculation"]
                    D3["Data Quality & Signal Confidence Index"]
                end

                subgraph DecisionSupport ["Educational Decision Support"]
                    E1["Sustained-Presence Attendance Marking"]
                    E2["AI Pedagogical Advisory Agent"]
                    E3["Interactive Analytics Dashboard"]
                    E4["Academic PDF & CSV Reports"]
                end

                A1 --> B1
                A2 --> B1
                A3 --> C1

                B1 --> B2 --> B3
                B3 --> B4 --> E1
                B3 --> B5 --> D1
                B3 --> B6 --> D1

                C1 --> C2 --> C3 --> C4 --> D1

                D1 --> D2 --> D3
                D2 --> E2
                D2 --> E3
                E1 --> E4
                E2 --> E4
            ```
            """
        )

    with tab_fusion:
        st.markdown("#### Transparent Multimodal Composite Scoring Formula")
        st.markdown(
            """
            The **Composite Activity Index ($C_t \\in [0.0, 1.0]$)** is calculated deterministically across synchronized discrete time windows ($T = 30\\text{s}$) 
            using the normalized weighted formulation:

            $$C_t = \\frac{w_{\\text{vis}} \\cdot V_t + w_{\\text{aud}} \\cdot A_t}{w_{\\text{vis}} + w_{\\text{aud}}}$$

            Where:
            - **$V_t$ (Visual Movement Index):** 
              $$V_t = 0.50 \\times \\min\\left(1.0, \\frac{\\bar{s}_t}{s_{\\text{active}}}\\right) + 0.50 \\times R_{\\text{forward}}$$
              Combines distance-normalized pixel speed $\\bar{s}_t$ and proportion of forward-facing head orientations $R_{\\text{forward}}$.
            - **$A_t$ (Speech Activity Ratio):** Proportion of acoustic sub-frames exceeding the speech energy threshold ($E \\ge 0.015$).
            - **$w_{\\text{vis}} = 0.35, w_{\\text{aud}} = 0.30$:** Configurable parameters in Settings.
            - **Data Quality Index ($Q_t$):** Measures the geometric reliability of facial landmark crops and audio stream presence.
            """
        )

    with tab_presence:
        st.markdown("#### Sustained-Presence Attendance State Machine")
        st.markdown(
            """
            Rather than marking attendance based on a single fleeting detection, AURORA enforces a multi-condition state machine:

            1. **Sustained Presence Evaluation:**
               $$P_{\\text{ratio}} = \\frac{T_{\\text{visible}}}{T_{\\text{session}}} \\ge 50\\%, \\quad T_{\\text{visible}} \\ge 30\\text{s}$$
            2. **Arrival Punctuality:**
               - **Present:** $T_{\\text{first}} < 0.20 \\times T_{\\text{session}}$ and $P_{\\text{ratio}} \\ge 50\\%$.
               - **Late:** $T_{\\text{first}} \\ge 0.20 \\times T_{\\text{session}}$ and $P_{\\text{ratio}} \\ge 50\\%$.
               - **Needs Review:** $30\\% \\le P_{\\text{ratio}} < 50\\%$ (indicates camera occlusion or partial attendance).
               - **Absent:** $P_{\\text{ratio}} < 30\\%$ or $T_{\\text{visible}} < 30\\text{s}$ (transient classroom visitor).
            3. **Anonymous Fallback:** Unmatched participants are tracked as `Anonymous Track #N` with zero identity fabrication.
            """
        )

    with tab_privacy:
        st.markdown("#### Privacy-Preserving Offline Guarantees")
        st.markdown(
            """
            1. **100% Air-Gapped Operation:** Video and audio feeds never leave the host system. No cloud APIs or telemetry.
            2. **Transient Image Buffers:** Raw video frames are discarded immediately after feature extraction; only numerical embeddings are stored for consented students.
            3. **Explicit Consent Flag:** Unenrolled participants are assigned anonymous identifiers (`Track #N`) and never falsely linked.
            4. **GDPR Right-to-Erasure:** Instructors can permanently wipe all biometric encodings with a single click in **Settings**.
            """
        )
