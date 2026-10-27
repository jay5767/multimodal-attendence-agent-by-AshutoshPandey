"""Video & Live Analysis page for video ingestion, metadata inspection, pipeline execution, and webcam streaming."""

import streamlit as st
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time

from config.settings import SAMPLES_DIR, DEFAULT_CONFIG
from database.repository import repo
from core.video_processor import VideoProcessingPipeline
from utils.sample_generator import generate_sample_classroom_video
from utils.camera_utils import get_default_camera_index

from ui.theme import render_aurora_hero, get_aurora_plotly_layout


def render_sessions() -> None:
    render_aurora_hero(
        title="VIDEO & LIVE ANALYSIS",
        subtitle="Ingest recorded classroom footage, inspect metadata, configure analysis rules, or capture live camera feeds",
        badge="REAL-TIME MULTIMODAL INGESTION"
    )

    tab_upload, tab_existing, tab_demo, tab_live = st.tabs([
        "📤 Upload & Process Lecture",
        "📂 Existing Sessions",
        "🧪 Synthetic Benchmark Generator",
        "📹 Live Camera & Mic Ingestion"
    ])

    # ========================================================
    # TAB 1: Upload & Process Video
    # ========================================================
    with tab_upload:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>New Classroom Video Session</h4>", unsafe_allow_html=True)
        
        c_meta1, c_meta2 = st.columns(2)
        with c_meta1:
            session_name = st.text_input("Session Title *", value=f"Lecture {datetime.now().strftime('%Y-%m-%d %H:%M')}", key="new_sess_title")
            courses = repo.list_courses()
            course_map = {c["id"]: f"{c['code']} - {c['name']}" for c in courses}
            course_id = st.selectbox("Course Assignment", options=list(course_map.keys()), format_func=lambda cid: course_map[cid]) if courses else None
        with c_meta2:
            instructor = st.text_input("Instructor / Lecturer Name", value="Prof. Elena Vance", key="new_sess_inst")
            session_date = st.date_input("Lecture Date", value=datetime.today(), key="new_sess_date")

        uploaded_file = st.file_uploader(
            "Select Recorded Classroom Video (MP4, AVI, MOV, MKV, WebM)",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help=f"Max size: {DEFAULT_CONFIG.video.max_file_size_mb} MB. Processed entirely locally."
        )

        # Video Metadata Inspector
        if uploaded_file is not None:
            # Save uploaded buffer to a temporary file to inspect metadata with OpenCV
            dest_dir = Path("data/videos")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / uploaded_file.name
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Read metadata
            v_cap = cv2.VideoCapture(str(dest_path))
            v_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = v_cap.get(cv2.CAP_PROP_FPS) or 25.0
            v_w = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_dur = v_frames / v_fps if v_fps > 0 else 0.0
            v_size_mb = round(uploaded_file.size / (1024 * 1024), 2)
            v_cap.release()

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 14px 18px; margin: 12px 0;">
                    <div style="color: #38bdf8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">VIDEO FILE METADATA</div>
                    <div style="display: flex; gap: 20px; font-size: 0.88rem; color: #cbd5e1; flex-wrap: wrap;">
                        <span>📁 <b>File:</b> {uploaded_file.name}</span>
                        <span>📦 <b>Size:</b> {v_size_mb} MB</span>
                        <span>⏱️ <b>Duration:</b> {format_seconds_to_mmss(v_dur)}</span>
                        <span>🖥️ <b>Resolution:</b> {v_w}x{v_h}</span>
                        <span>🎞️ <b>FPS:</b> {v_fps:.1f}</span>
                        <span>🖼️ <b>Total Frames:</b> {v_frames}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Analysis Configuration Accordion
            with st.expander("⚙️ Advanced Pipeline Configuration (Optional)", expanded=False):
                cfg_c1, cfg_c2 = st.columns(2)
                with cfg_c1:
                    cust_fps = st.slider("Visual Frame Sampling Rate (FPS)", min_value=1.0, max_value=10.0, value=2.0, step=0.5,
                                         help="Higher FPS provides more frequent pose tracking at the cost of processing time.")
                    min_pres = st.slider("Sustained Presence Threshold (% of duration)", min_value=10, max_value=90, value=50, step=5,
                                         help="Participants visible for at least this percentage are marked Present.")
                with cfg_c2:
                    face_conf = st.slider("Face Match Similarity Threshold", min_value=0.30, max_value=0.85, value=0.55, step=0.05,
                                          help="Cosine similarity required to match tracked person to enrolled student.")
                    audio_sens = st.slider("Speech RMS Energy Threshold", min_value=0.005, max_value=0.050, value=0.015, step=0.005,
                                           help="Threshold above which sound energy is classified as speech activity.")

            if st.button("🚀 Start Multimodal Processing Pipeline", type="primary", use_container_width=True):
                # Update runtime config
                DEFAULT_CONFIG.video.sample_rate_fps = cust_fps
                DEFAULT_CONFIG.attendance.min_presence_ratio = min_pres / 100.0
                DEFAULT_CONFIG.attendance.face_similarity_threshold = face_conf
                DEFAULT_CONFIG.engagement.energy_speech_threshold = audio_sens

                session_id = repo.create_session(
                    session_name=session_name,
                    course_id=course_id,
                    instructor=instructor,
                    date=str(session_date),
                    video_path=str(dest_path),
                    fps=v_fps,
                    resolution=f"{v_w}x{v_h}"
                )

                st.markdown(f"**Session #{session_id} Initialized.** Executing pipeline...")
                progress_bar = st.progress(0.0)
                status_box = st.empty()
                preview_box = st.empty()

                def on_progress(fraction: float, message: str):
                    progress_bar.progress(min(1.0, max(0.0, fraction)))
                    status_box.markdown(f"**Stage:** `{message}`")

                def on_preview(frame, meta):
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_box.image(
                        rgb_frame,
                        caption=f"Frame: {meta.get('frame', 0)} | Timestamp: {meta.get('timestamp', 0):.1f}s | Active Tracks: {meta.get('active_count', 0)}",
                        use_container_width=True
                    )

                pipeline = VideoProcessingPipeline(
                    config=DEFAULT_CONFIG,
                    progress_callback=on_progress,
                    frame_preview_callback=on_preview
                )

                try:
                    res = pipeline.process_session(session_id, dest_path)
                    st.balloons()
                    st.success(f"✅ Processing complete! Tracked {res.get('attendance_count', 0)} participants and generated {res.get('recommendations_count', 0)} teaching suggestions.")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline execution encountered an error: {e}")

    # ========================================================
    # TAB 2: Existing Sessions
    # ========================================================
    with tab_existing:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Stored Classroom Sessions</h4>", unsafe_allow_html=True)
        sessions = repo.list_sessions()
        if not sessions:
            st.info("No recorded sessions found in database.")
        else:
            df_sess = pd.DataFrame(sessions)
            df_sess["Duration"] = df_sess["duration_seconds"].apply(format_seconds_to_mmss)
            cols = ["id", "session_name", "course_name", "instructor", "date", "Duration", "status", "resolution", "fps"]
            st.dataframe(df_sess[[c for c in cols if c in df_sess.columns]], use_container_width=True)

            st.markdown("<h5 style='color: #f8fafc; font-size: 0.95rem; font-weight: 700; margin-top: 14px;'>Session Actions</h5>", unsafe_allow_html=True)
            sel_id = st.selectbox(
                "Select Session to Manage:",
                options=[s["id"] for s in sessions],
                format_func=lambda sid: f"Session #{sid} • {next(s['session_name'] for s in sessions if s['id'] == sid)}",
                key="manage_sess_select"
            )

            if sel_id:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔄 Reprocess Session Pipeline", use_container_width=True):
                        s_info = next(s for s in sessions if s["id"] == sel_id)
                        vpath = s_info.get("video_path")
                        if vpath and Path(vpath).exists():
                            p_bar = st.progress(0.0)
                            s_text = st.empty()
                            p_line = VideoProcessingPipeline(
                                progress_callback=lambda f, m: (p_bar.progress(min(1.0, f)), s_text.write(m))
                            )
                            p_line.process_session(sel_id, Path(vpath))
                            st.success(f"Reprocessed Session #{sel_id} successfully!")
                            st.rerun()
                        else:
                            st.error("Original video file not found on disk at specified path.")
                with col_btn2:
                    if st.button("❌ Delete Session Record & Analytics", type="primary", use_container_width=True):
                        repo.delete_session(sel_id)
                        st.warning(f"Session #{sel_id} permanently removed.")
                        st.rerun()

    # ========================================================
    # TAB 3: Synthetic Benchmark Generator
    # ========================================================
    with tab_demo:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Generate Synthetic Classroom Benchmark Video</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='color: #94a3b8; font-size: 0.9rem;'>
            Use this built-in benchmark generator to test AURORA's entire pipeline without needing an external video file.
            Renders an instructional classroom lecture with 3 student avatars (demonstrating forward gaze, looking down, and turning away),
            whiteboard text, and a synchronized 16kHz speech audio track.
            </p>
            """,
            unsafe_allow_html=True
        )

        demo_dur = st.slider("Benchmark Video Duration (Seconds)", min_value=10, max_value=60, value=15, step=5, key="bench_dur_slider")

        if st.button("⚡ Generate & Analyze Synthetic Classroom Video", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic classroom video and 16kHz PCM audio..."):
                sample_file = SAMPLES_DIR / f"classroom_benchmark_{demo_dur}s.mp4"
                generate_sample_classroom_video(sample_file, duration_seconds=demo_dur, fps=15)

            st.success(f"Benchmark video rendered: `{sample_file.name}`")

            # Create session for it
            c_list = repo.list_courses()
            c_id = c_list[0]["id"] if c_list else repo.create_course("CS302", "Multimodal Classroom AI", "Prof. Vance")
            sess_id = repo.create_session(
                session_name=f"CS302 Synthetic Benchmark ({demo_dur}s)",
                course_id=c_id,
                instructor="Prof. Elena Vance",
                date=datetime.now().strftime("%Y-%m-%d"),
                video_path=str(sample_file)
            )

            p_bar = st.progress(0.0)
            s_text = st.empty()
            prev_view = st.empty()

            pipeline = VideoProcessingPipeline(
                progress_callback=lambda f, m: (p_bar.progress(min(1.0, f)), s_text.write(m)),
                frame_preview_callback=lambda f, m: prev_view.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), caption=f"Time: {m.get('timestamp',0):.1f}s | Active Tracks: {m.get('active_count', 0)}", use_container_width=True)
            )

            res = pipeline.process_session(sess_id, sample_file)
            st.balloons()
            st.success(f"✅ Benchmark session #{sess_id} processed! View comprehensive results in Overview Dashboard or Attendance Management.")
            time.sleep(1.5)
            st.rerun()

    # ========================================================
    # TAB 4: Live Camera & Microphone Ingestion
    # ========================================================
    with tab_live:
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.1rem; font-weight: 700;'>Live Camera & Microphone Input</h4>", unsafe_allow_html=True)
        st.caption("Acquire live video directly from your connected webcam for instantaneous participant tracking or live session recording.")

        live_mode = st.radio(
            "Select Live Mode:",
            ["📸 Instant Attendance Snapshot", "🎥 Record Live Webcam Session"],
            horizontal=True,
            key="live_input_mode_radio"
        )

        if live_mode == "📸 Instant Attendance Snapshot":
            st.info("Capture a live frame from your webcam to immediately detect participants, estimate head pose, and match against enrolled student templates.")
            live_snapshot = st.camera_input("Take Live Classroom Snapshot")

            if live_snapshot is not None:
                bytes_data = live_snapshot.getvalue()
                nparr = np.frombuffer(bytes_data, np.uint8)
                live_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                from core.person_detector import PersonDetector
                from core.face_recognition import FaceRecognizer
                from core.head_pose import HeadPoseEstimator

                detector = PersonDetector()
                recognizer = FaceRecognizer()
                students = repo.list_students(active_only=True)
                recognizer.load_enrolled_students(students)
                pose_est = HeadPoseEstimator()

                dets = detector.detect(live_frame)
                st.markdown(f"**Detected {len(dets)} participant(s)** in webcam snapshot.")

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

                    # Head pose
                    pose = pose_est.estimate(live_frame, p_box, track_id=idx + 1, timestamp_sec=0.0)

                    label = f"{matched_sid} ({int(conf*100)}%)" if matched_sid else f"Anonymous Track #{idx+1}"
                    color = (16, 185, 129) if matched_sid else (56, 189, 248)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    with cols[col_idx % len(cols)]:
                        st.markdown(
                            f"""
                            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                                <div style="font-weight: 700; color: #f8fafc;">Participant #{idx+1}</div>
                                <div style="font-size: 0.85rem; color: #38bdf8;">Identity: <b>{matched_sid or 'Anonymous Track'}</b></div>
                                <div style="font-size: 0.82rem; color: #cbd5e1;">Pose: {pose.observable_state}</div>
                                <div style="font-size: 0.82rem; color: #94a3b8;">Conf: {int(conf*100)}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    col_idx += 1

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Live Detection & Facial Recognition Overlay", use_container_width=True)

        elif live_mode == "🎥 Record Live Webcam Session":
            def_idx = get_default_camera_index()
            st.info(f"Record live classroom footage from your connected webcam device (Auto-detected: #{def_idx}) and process it through the full multimodal pipeline.")
            
            c_rec1, c_rec2 = st.columns(2)
            with c_rec1:
                live_sess_name = st.text_input("Live Session Name", value=f"Live Session {datetime.now().strftime('%H:%M:%S')}")
                cam_index = st.number_input("Camera Device Index", min_value=0, max_value=5, value=int(def_idx), step=1)
            with c_rec2:
                rec_duration = st.slider("Recording Duration (Seconds)", min_value=5, max_value=30, value=10, step=5)

            if st.button("🔴 Start Live Webcam Recording & Processing", type="primary", use_container_width=True):
                cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(cam_index))
                if not cap.isOpened():
                    st.error(f"Cannot access camera device {cam_index}. Please check device permissions.")
                else:
                    live_dir = Path("data/videos")
                    live_dir.mkdir(parents=True, exist_ok=True)
                    live_path = live_dir / f"webcam_capture_{int(time.time())}.mp4"

                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    fps = 15.0
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(live_path), fourcc, fps, (w, h))

                    st.warning(f"Recording live for {rec_duration} seconds... Please look at camera.")
                    rec_progress = st.progress(0.0)
                    start_time = time.time()

                    while time.time() - start_time < rec_duration:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        elapsed = time.time() - start_time
                        rec_progress.progress(min(1.0, elapsed / rec_duration))
                        time.sleep(1.0 / fps)

                    cap.release()
                    out.release()

                    st.success(f"Webcam footage recorded ({rec_duration}s) to `{live_path.name}`! Initializing pipeline...")

                    c_list = repo.list_courses()
                    c_id = c_list[0]["id"] if c_list else repo.create_course("LIVE-101", "Live Classroom Stream", "Instructor")
                    s_id = repo.create_session(
                        session_name=live_sess_name,
                        course_id=c_id,
                        instructor="Webcam Stream",
                        date=datetime.now().strftime("%Y-%m-%d"),
                        video_path=str(live_path)
                    )

                    pipeline = VideoProcessingPipeline()
                    pipeline.process_session(s_id, live_path)
                    st.balloons()
                    st.success(f"Live session #{s_id} analyzed successfully!")
                    time.sleep(1.5)
                    st.rerun()
