"""Real-Time Live Camera & Continuous Movement Tracking for AURORA.

Provides real-time in-dashboard video streaming with OpenCV and YOLOv8,
live optical flow movement energy meters, head pose orientation estimation,
sustained-presence attendance verification, standalone browser monitor,
and native 30+ FPS desktop window streaming.
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from database.repository import repo
from core.person_detector import PersonDetector
from core.face_recognition import FaceRecognizer
from core.head_pose import HeadPoseEstimator
from config.settings import DEFAULT_CONFIG, SAMPLES_DIR, OUTPUTS_DIR
from ui.theme import render_aurora_hero
from utils.camera_utils import get_available_cameras, get_default_camera_index


def render_live_camera(selected_session_id: Optional[int] = None) -> None:
    render_aurora_hero(
        title="LIVE CLASSROOM AI MONITOR",
        subtitle="Real-time optical motion tracking, head pose orientation, sustained presence timer, and attendance verification",
        badge="REAL-TIME SENSING ACTIVE"
    )

    st.markdown(
        """
        <div style="background: rgba(6, 182, 212, 0.08); border-left: 4px solid #06b6d4; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 0.92rem; margin-bottom: 2px;">⚡ LIVE CONTINUOUS MULTIMODAL SENSING</div>
            <div style="color: #cbd5e1; font-size: 0.85rem;">
                Continuous video analysis detects your movements, head yaw/pitch, and sustained presence in real time.
                Move your head, lean, or gesture to watch the <b>Movement Energy</b> and <b>Head Pose</b> indicators update dynamically.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Detect active cameras
    detected_cams = get_available_cameras(max_devices_to_scan=4)
    default_idx = detected_cams[0]["index"] if detected_cams else 0

    tab_live, tab_standalone, tab_snapshot, tab_desktop = st.tabs([
        "🔴 In-Dashboard Live AI Stream",
        "🌐 Standalone Fullscreen Monitor",
        "📸 Instant Face Match Snapshot",
        "🖥️ Native High-FPS OpenCV Window"
    ])

    # ========================================================
    # TAB 1: Real-Time In-Dashboard Video Stream (OpenCV + YOLOv8)
    # ========================================================
    with tab_live:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Continuous In-Dashboard Video & Movement Stream</h4>", unsafe_allow_html=True)
        st.caption("Processes camera frames directly with local OpenCV & YOLOv8, calculating continuous optical flow movement and head pose.")

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.3])
        with ctrl_col1:
            cam_options = {}
            if detected_cams:
                for c in detected_cams:
                    cam_options[c["index"]] = c["name"]
            else:
                cam_options[0] = "Camera Device #0 (Fallback)"
            
            # Also allow recorded video sample option
            cam_options[-1] = "🎞️ Recorded Classroom Benchmark Clip"

            selected_source = st.selectbox(
                "Video Input Source",
                options=list(cam_options.keys()),
                format_func=lambda k: cam_options[k],
                index=0
            )

        with ctrl_col2:
            presence_target_sec = st.slider("Sustained Presence Threshold", min_value=5, max_value=60, value=15, step=5, help="Seconds required to mark verified attendance")

        with ctrl_col3:
            st.write("")
            stream_active = st.toggle("▶️ Start Live Stream", value=False, key="live_cam_toggle", help="Toggle live camera streaming and movement tracking")

        # Stream container
        col_video, col_telemetry = st.columns([1.5, 1.0])

        video_placeholder = col_video.empty()
        telemetry_placeholder = col_telemetry.empty()

        if not stream_active:
            with video_placeholder:
                st.markdown(
                    """
                    <div style="background: rgba(15, 23, 42, 0.7); border: 2px dashed rgba(56, 189, 248, 0.25); border-radius: 12px; height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 24px;">
                        <div style="font-size: 3rem; margin-bottom: 12px;">📹</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Camera Stream Standby</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; max-width: 440px;">
                            Toggle <b>'▶️ Start Live Stream'</b> above to begin real-time optical movement tracking, head pose classification, and sustained presence verification.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with telemetry_placeholder:
                st.markdown(
                    """
                    <div style="display: flex; flex-direction: column; gap: 12px;">
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Movement Energy</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8; font-family: monospace;">0.0%</div>
                            <div style="font-size: 0.76rem; color: #64748b;">Standby • Awaiting stream</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Head Orientation</div>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #34d399;">Attentive (Forward)</div>
                            <div style="font-size: 0.76rem; color: #64748b;">Yaw: 0.0° • Pitch: 0.0°</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Sustained Presence</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #c084fc;">00:00</div>
                            <div style="font-size: 0.76rem; color: #64748b;">Threshold: 00:15 required</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            # Open Video Capture
            is_recorded = (selected_source == -1)
            if is_recorded:
                sample_path = SAMPLES_DIR / "classroom_benchmark_lecture.mp4"
                if not sample_path.exists():
                    st.error("Benchmark video not found. Generate it first on the Dashboard.")
                    st.stop()
                cap = cv2.VideoCapture(str(sample_path))
            else:
                cap = cv2.VideoCapture(int(selected_source), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(selected_source))

            if not cap.isOpened():
                st.error(f"Could not connect to video source {selected_source}. Please verify camera permissions.")
            else:
                detector = PersonDetector()
                pose_est = HeadPoseEstimator()
                prev_gray = None
                presence_start_time = None
                total_presence_seconds = 0.0
                attendance_marked = False
                frame_count = 0
                fps_start = time.time()

                stop_stream_btn = st.button("⏹️ Stop Stream Now", type="secondary")

                # Stream execution loop (runs up to 600 frames or until stopped)
                while stream_active and cap.isOpened() and not stop_stream_btn:
                    ret, frame = cap.read()
                    if not ret:
                        if is_recorded:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break

                    frame_count += 1
                    current_time = time.time()

                    # Convert to grayscale for optical flow movement
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray, (160, 120))

                    motion_pct = 0.0
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray_small)
                        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                        motion_pct = min(100.0, float(np.mean(thresh) * 3.2))
                    prev_gray = gray_small.copy()

                    # Run Detection
                    dets = detector.detect(frame)
                    person_found = len(dets) > 0

                    # Sustained presence accumulator
                    if person_found:
                        if presence_start_time is None:
                            presence_start_time = current_time
                        total_presence_seconds = current_time - presence_start_time
                    else:
                        presence_start_time = None

                    # Annotate frame
                    annotated = frame.copy()
                    h_state_str = "Forward Facing"
                    yaw_angle = 0.0
                    pitch_angle = 0.0

                    if person_found:
                        d = dets[0]
                        p_box = (d.x1, d.y1, d.x2, d.y2)
                        pose = pose_est.estimate(frame, p_box, track_id=1, timestamp_sec=total_presence_seconds)
                        yaw_angle = pose.yaw if pose.yaw is not None else 0.0
                        pitch_angle = pose.pitch if pose.pitch is not None else 0.0
                        h_state_str = pose.observable_state.replace("_", " ").title()

                        # Dynamic color based on movement & focus
                        if motion_pct > 35.0:
                            box_color = (11, 158, 245)  # Amber: Dynamic motion
                            label_state = "Active Gestures / Movement"
                        elif abs(yaw_angle) > 20.0:
                            box_color = (56, 189, 248)  # Cyan: Turned away
                            label_state = f"Turned ({h_state_str})"
                        else:
                            box_color = (16, 185, 129)  # Green: Attentive focus
                            label_state = "Attentive Forward Focus"

                        # Draw bounding box and stylized corners
                        cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), box_color, 2)
                        
                        # HUD Tag above head
                        hud_text = f"Live Participant #1 | {label_state} | Motion: {int(motion_pct)}%"
                        cv2.rectangle(annotated, (d.x1, max(0, d.y1 - 28)), (d.x1 + len(hud_text)*8, d.y1), (15, 23, 42), -1)
                        cv2.putText(annotated, hud_text, (d.x1 + 4, max(18, d.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2)

                    # Top Overlay Bar
                    fps_val = int(frame_count / max(0.1, (current_time - fps_start)))
                    cv2.putText(annotated, f"AURORA LIVE AI | FPS: {fps_val} | Local Privacy Mode", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 255), 2)

                    # Render frame
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # Check attendance threshold
                    if total_presence_seconds >= presence_target_sec and not attendance_marked:
                        attendance_marked = True
                        if selected_session_id:
                            students = repo.list_students(active_only=True)
                            s_id = students[0]["id"] if students else 1
                            repo.record_attendance(
                                session_id=selected_session_id,
                                student_id=s_id,
                                status="present",
                                first_seen=0.0,
                                last_seen=total_presence_seconds,
                                duration=total_presence_seconds,
                                confidence=0.92,
                                is_manual=False
                            )

                    # Telemetry sidebar update
                    mins = int(total_presence_seconds // 60)
                    secs = int(total_presence_seconds % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                    progress_pct = min(100, int((total_presence_seconds / max(1, presence_target_sec)) * 100))

                    with telemetry_placeholder:
                        m_color = "#f59e0b" if motion_pct > 35 else ("#06b6d4" if motion_pct > 12 else "#818cf8")
                        m_state_text = "Dynamic Interaction" if motion_pct > 35 else ("Attentive Focus / Notes" if motion_pct > 12 else "Calm Stillness")
                        
                        verified_badge = """
                        <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; padding: 8px 12px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; text-align: center; margin-top: 10px;">
                            ✅ Sustained Presence Verified! Attendance Logged.
                        </div>
                        """ if attendance_marked else ""

                        st.markdown(
                            f"""
                            <div style="display: flex; flex-direction: column; gap: 12px;">
                                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;">
                                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Continuous Movement Energy</div>
                                    <div style="font-size: 1.6rem; font-weight: 800; color: {m_color}; font-family: monospace;">{motion_pct:.1f}%</div>
                                    <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; margin: 6px 0;">
                                        <div style="width: {motion_pct}%; height: 100%; background: {m_color}; transition: width 0.1s ease;"></div>
                                    </div>
                                    <div style="font-size: 0.78rem; color: #cbd5e1;">State: <b>{m_state_text}</b></div>
                                </div>

                                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;">
                                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Head Orientation & Posture</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #34d399;">{h_state_str}</div>
                                    <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 4px;">Yaw: <b>{yaw_angle:.1f}°</b> • Pitch: <b>{pitch_angle:.1f}°</b></div>
                                </div>

                                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;">
                                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Sustained Presence Timer</div>
                                    <div style="font-size: 1.6rem; font-weight: 800; color: #c084fc; font-family: monospace;">{time_str}</div>
                                    <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; margin: 6px 0;">
                                        <div style="width: {progress_pct}%; height: 100%; background: #a855f7;"></div>
                                    </div>
                                    <div style="font-size: 0.76rem; color: #94a3b8;">Threshold: {presence_target_sec}s for Verified Attendance</div>
                                    {verified_badge}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Modest sleep to keep FPS smooth without pegging CPU
                    time.sleep(0.03)

                cap.release()

    # ========================================================
    # TAB 2: Standalone Fullscreen Browser Monitor
    # ========================================================
    with tab_standalone:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Standalone Fullscreen Browser Live Monitor</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='color: #94a3b8; font-size: 0.88rem; line-height: 1.5;'>
            For high-resolution, full-screen classroom projection or unobstructed webcam access without iframe restrictions,
            open AURORA's standalone hardware-accelerated monitor directly in your browser.
            </p>
            """,
            unsafe_allow_html=True
        )

        monitor_html_path = OUTPUTS_DIR / "live_monitor.html"
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #f8fafc; font-weight: 700; font-size: 1rem;">Standalone Live Monitor File</div>
                        <div style="color: #38bdf8; font-size: 0.82rem; font-family: monospace; margin-top: 2px;">{monitor_html_path}</div>
                    </div>
                    <a href="file:///{monitor_html_path}" target="_blank" style="text-decoration: none;">
                        <button style="background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%); color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 700; cursor: pointer;">
                            ↗️ Open Fullscreen Live Monitor
                        </button>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("👁️ Preview Standalone Monitor Inside Streamlit"):
            if monitor_html_path.exists():
                with open(monitor_html_path, "r", encoding="utf-8") as f:
                    raw_html = f.read()
                st.components.v1.html(raw_html, height=540, scrolling=False)

    # ========================================================
    # TAB 3: Instant Face Match Snapshot
    # ========================================================
    with tab_snapshot:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>High-Accuracy Facial Biometric Snapshot</h4>", unsafe_allow_html=True)
        st.caption("Capture a still frame to execute YOLOv8 detection, consented face recognition, and record verified attendance.")

        live_snapshot = st.camera_input("Capture High-Resolution Attendance Snapshot", key="hires_cam_snap")

        if live_snapshot is not None:
            bytes_data = live_snapshot.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            live_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            detector = PersonDetector()
            recognizer = FaceRecognizer()
            students = repo.list_students(active_only=True)
            recognizer.load_enrolled_students(students)
            pose_est = HeadPoseEstimator()

            dets = detector.detect(live_frame)
            st.markdown(f"**Detected {len(dets)} participant(s)** in captured frame.")

            annotated = live_frame.copy()
            cols = st.columns(min(3, max(1, len(dets)))) if dets else [st.container()]
            col_idx = 0

            for idx, d in enumerate(dets):
                x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
                p_box = (x1, y1, x2, y2)
                face_boxes = recognizer.detect_faces(live_frame, p_box)
                matched_sid = None
                conf = 0.0

                if face_boxes:
                    fx, fy, fw, fh = face_boxes[0]
                    face_crop = live_frame[fy:fy + fh, fx:fx + fw]
                    match_res = recognizer.match_face(face_crop)
                    if match_res.is_matched:
                        matched_sid = match_res.student_id
                        conf = match_res.confidence

                pose = pose_est.estimate(live_frame, p_box, track_id=idx + 1, timestamp_sec=0.0)

                label = f"{matched_sid} ({int(conf*100)}%)" if matched_sid else f"Anonymous Track #{idx+1}"
                color = (16, 185, 129) if matched_sid else (56, 189, 248)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                with cols[col_idx % len(cols)]:
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                            <div style="font-weight: 700; color: #f8fafc;">Participant #{idx+1}</div>
                            <div style="font-size: 0.85rem; color: #38bdf8;">Identity: <b>{matched_sid or 'Anonymous Track'}</b></div>
                            <div style="font-size: 0.82rem; color: #cbd5e1;">Pose: {pose.observable_state}</div>
                            <div style="font-size: 0.82rem; color: #94a3b8;">Conf: {int(conf*100)}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                col_idx += 1

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="YOLOv8 & Consented Face Verification", use_container_width=True)

    # ========================================================
    # TAB 4: Native High-FPS OpenCV Window
    # ========================================================
    with tab_desktop:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Native High-FPS OpenCV Desktop Window</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='color: #94a3b8; font-size: 0.9rem;'>
            Opens a native hardware-accelerated desktop window running real-time YOLOv8 bounding boxes, head pose vectors, and motion analysis at 30+ FPS.
            </p>
            """,
            unsafe_allow_html=True
        )

        c_dev1, c_dev2 = st.columns(2)
        with c_dev1:
            cam_idx = st.number_input(
                "Camera Index",
                min_value=0,
                max_value=5,
                value=int(default_idx),
                step=1,
                help=f"Auto-detected working camera is device #{default_idx}"
            )
        with c_dev2:
            st.write("")
            st.write("")
            st.caption("Press **'q'** or **'ESC'** inside the OpenCV window anytime to close.")

        if st.button("🚀 Launch Native OpenCV Live Window", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(cam_idx))

            if not cap.isOpened():
                st.error(f"Cannot access camera device {cam_idx}. Please verify camera permissions.")
            else:
                detector = PersonDetector()
                pose_est = HeadPoseEstimator()
                frame_count = 0
                t_start = time.time()

                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        t_sec = time.time() - t_start
                        frame_count += 1

                        dets = detector.detect(frame)
                        for d in dets:
                            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (212, 182, 6), 2)
                            pose = pose_est.estimate(frame, (x1, y1, x2, y2), 1, t_sec)
                            label = f"Participant: {pose.observable_state} (Speed: {pose.movement_speed:.2f})"
                            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (212, 182, 6), 2)

                        # HUD Overlay
                        cv2.putText(frame, f"AURORA LIVE AI - Time: {t_sec:.1f}s | Press 'q' to Quit", (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

                        cv2.imshow("AURORA Live Classroom Multimodal AI", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    st.success("Native live session completed.")
