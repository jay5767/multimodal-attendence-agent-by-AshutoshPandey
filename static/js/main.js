/**
 * Multimodal Attendance & Engagement Agent
 * Complete Single Page Application (SPA) Controller
 * Features: Zero-Reload Tab Transitions, Smart Hardware Camera Detection, 
 * Server Direct Stream Mode, Bounding Box Overlays, Circular Gauge, Dynamic Data Ingestion.
 */

let socket = null;
let videoElement = null;
let serverCamElement = null;
let overlayCanvas = null;
let overlayCtx = null;
let hiddenCanvas = null;
let hiddenCtx = null;
let cameraStream = null;
window.cameraStream = null;
let isStreaming = false;
let isCameraPaused = false;
let isServerCamActive = false;
let isAnonymousMode = false;
let sessionTimerInterval = null;
let sessionStartTimestamp = null;
let dropCount = 0;
let preferredDeviceId = null;

// Circular Gauge Constants (r=80, circumference = 2 * PI * 80 ~= 502.65)
const GAUGE_CIRCUMFERENCE = 502.65;

document.addEventListener("DOMContentLoaded", () => {
  initSocket();
  initMedia();
  initBrowserAudio();
  initAnonymousState();
  initConsentState();

  // Initialize SPA view based on URL or server-provided initial_view
  const path = window.location.pathname.replace(/^\//, "") || window.INITIAL_VIEW || "live";
  switchView(path, false);

  // Handle browser back / forward navigation seamlessly
  window.addEventListener("popstate", (e) => {
    const p = window.location.pathname.replace(/^\//, "") || "live";
    switchView(p, false);
  });
});

// ============================================================================
// 1. SINGLE PAGE APPLICATION (SPA) ZERO-RELOAD VIEW SWITCHING
// ============================================================================
function switchView(viewName, pushHistory = true) {
  if (!viewName || viewName === "") viewName = "live";

  // Hide all view containers
  const views = document.querySelectorAll(".app-view");
  views.forEach(v => v.classList.remove("active"));

  // Deactivate all nav links
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach(l => l.classList.remove("active"));

  // Activate target view
  const targetView = document.getElementById(`view-${viewName}`);
  const targetNav = document.getElementById(`nav-item-${viewName}`);

  if (targetView) {
    targetView.classList.add("active");
  } else {
    // Fallback to live view
    const liveView = document.getElementById("view-live");
    if (liveView) liveView.classList.add("active");
    viewName = "live";
  }

  if (targetNav) {
    targetNav.classList.add("active");
  }

  // Update URL history without page reload
  if (pushHistory) {
    const stateUrl = viewName === "live" ? "/" : `/${viewName}`;
    if (window.location.pathname !== stateUrl) {
      window.history.pushState({ view: viewName }, "", stateUrl);
    }
  }

  // Trigger dynamic data fetch for the active view
  switch (viewName) {
    case "enrollment":
      fetchEnrolledStudents();
      break;
    case "attendance":
      fetchFullAttendance();
      break;
    case "history":
      fetchSessionHistory();
      break;
    case "insights":
      fetchInsightsTimeline();
      break;
    case "timetable":
      if (typeof fetchTimetableData === "function") fetchTimetableData();
      break;
    case "analytics":
      // Resize Chart.js to fit the container
      if (window.engagementChart) window.engagementChart.resize();
      if (window.audioChart) window.audioChart.resize();
      break;
    case "live":
      // Refresh camera layout and ensure tracks are active
      adjustCanvasDimensions();
      ensureCameraActive();
      if (window.dashboardChart) window.dashboardChart.resize();
      break;
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ============================================================================
// 2. SOCKET.IO INITIALIZATION
// ============================================================================
function initSocket() {
  socket = io();

  socket.on("connect", () => {
    console.log("🟢 Connected to AI Classroom backend via Gevent WebSocket");
    const statusBanner = document.getElementById("cam-status-msg");
    if (statusBanner) statusBanner.innerText = "Real-time edge connection established.";
  });

  socket.on("frame_processed", (data) => {
    handleProcessedFrame(data);
  });

  socket.on("interval_completed", (intervalData) => {
    console.log("⏱️ Interval completed:", intervalData);
    if (intervalData.is_drop) {
      dropCount++;
      const el = document.getElementById("stat-drop-count");
      if (el) el.innerText = dropCount;
    }
  });

  socket.on("new_suggestion", (sugg) => {
    displayNewSuggestion(sugg);
  });

  socket.on("session_state_changed", (data) => {
    handleSessionStateChange(data);
  });
}

// ============================================================================
// 3. SMART CAMERA INGESTION (HARDWARE AUTO-DETECTION & SERVER DIRECT CAPTURE)
// ============================================================================
async function initMedia() {
  videoElement = document.getElementById("webcam-video");
  serverCamElement = document.getElementById("server-cam-feed");
  overlayCanvas = document.getElementById("overlay-canvas");
  if (!videoElement || !overlayCanvas) return;

  overlayCtx = overlayCanvas.getContext("2d");

  hiddenCanvas = document.createElement("canvas");
  hiddenCanvas.width = 640;
  hiddenCanvas.height = 480;
  hiddenCtx = hiddenCanvas.getContext("2d");

  // Populate camera select with physical hardware inspection
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    const camSelect = document.getElementById("camera-select");

    if (camSelect) {
      camSelect.innerHTML = "";

      // Add direct server OpenCV option
      const serverOpt = document.createElement("option");
      serverOpt.value = "server_direct";
      serverOpt.text = "⚡ Direct Server Hardware (Index 2, DSHOW)";
      camSelect.appendChild(serverOpt);

      let bestDevice = null;
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement("option");
        opt.value = dev.deviceId;
        const label = dev.label || `Camera ${idx + 1}`;
        opt.text = `📹 ${label}`;
        camSelect.appendChild(opt);

        const lower = label.toLowerCase();
        // Identify integrated physical webcam
        if (
          !bestDevice &&
          (lower.includes("integrated") || lower.includes("hd") || lower.includes("webcam") || lower.includes("camera")) &&
          !lower.includes("iriun") && !lower.includes("voicemod") && !lower.includes("virtual")
        ) {
          bestDevice = dev.deviceId;
        }
      });

      if (bestDevice) {
        preferredDeviceId = bestDevice;
      } else if (videoDevices.length > 0) {
        preferredDeviceId = videoDevices[0].deviceId;
      }

      camSelect.addEventListener("change", () => {
        if (camSelect.value === "server_direct") {
          setServerCameraMode(true);
        } else {
          setServerCameraMode(false);
          startCamera(camSelect.value);
        }
      });
    }

    // Default to browser camera
    startCamera(preferredDeviceId);
  } catch (err) {
    console.warn("Device enumeration error:", err);
    startCamera();
  }
}

async function startCamera(deviceId = null) {
  if (isServerCamActive) return;

  if (cameraStream) {
    try {
      cameraStream.getTracks().forEach(t => t.stop());
    } catch (e) {}
    cameraStream = null;
    window.cameraStream = null;
  }

  const constraints = deviceId
    ? { video: { deviceId: { ideal: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }
    : { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false };

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    window.cameraStream = cameraStream;
    if (videoElement) {
      videoElement.srcObject = cameraStream;
      videoElement.style.display = "block";
    }
    if (serverCamElement) serverCamElement.style.display = "none";

    const errBox = document.getElementById("camera-error");
    if (errBox) errBox.style.display = "none";

    const banner = document.getElementById("cam-status-msg");
    if (banner) banner.innerText = "Browser camera stream active. Evaluating sustained presence.";

    const badge = document.getElementById("cam-active-source-badge");
    if (badge) badge.innerText = "Integrated Cam (Browser)";

    if (videoElement) {
      videoElement.play().catch(e => console.warn("Video play notice:", e));
    }
    isStreaming = true;
    adjustCanvasDimensions();
    requestAnimationFrame(captureAndSendLoop);
  } catch (err) {
    console.warn("Camera initialization notice:", err);
    if (deviceId) {
      return startCamera(null);
    }
    const errBox = document.getElementById("camera-error");
    if (errBox) {
      errBox.innerText = `Camera notice: ${err.message}. Please check browser camera permissions.`;
      errBox.style.display = "block";
    }
  }
}

function ensureCameraActive() {
  if (isServerCamActive) return;
  const isDead = !cameraStream || !cameraStream.active || 
    cameraStream.getVideoTracks().length === 0 || 
    cameraStream.getVideoTracks().some(t => t.readyState === "ended");

  if (isDead) {
    startCamera(preferredDeviceId);
  } else if (videoElement && videoElement.paused) {
    videoElement.play().catch(e => console.warn("Resume play notice:", e));
    requestAnimationFrame(captureAndSendLoop);
  }
}

function adjustCanvasDimensions() {
  if (!overlayCanvas || !videoElement) return;
  overlayCanvas.width = videoElement.videoWidth || 640;
  overlayCanvas.height = videoElement.videoHeight || 480;
  if (hiddenCanvas) {
    hiddenCanvas.width = 640;
    hiddenCanvas.height = 480;
  }
}

async function toggleServerCameraMode() {
  setServerCameraMode(!isServerCamActive);
}

async function setServerCameraMode(enableServer) {
  isServerCamActive = enableServer;
  const btn = document.getElementById("btn-switch-server-cam");
  const badge = document.getElementById("cam-active-source-badge");
  const banner = document.getElementById("cam-status-msg");

  if (isServerCamActive) {
    // Stop browser camera tracks
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
    if (videoElement) videoElement.style.display = "none";
    if (serverCamElement) {
      serverCamElement.src = "/video_feed?" + Date.now();
      serverCamElement.style.display = "block";
    }

    if (btn) btn.innerText = "Switch to Browser Cam";
    if (badge) badge.innerText = "Server Direct Cam (Index 2 DSHOW)";
    if (banner) banner.innerText = "Server-side OpenCV Direct Capture active (Camera 2 @ 60 FPS).";

    // Notify backend
    try {
      await fetch("/api/camera/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "server" })
      });
    } catch (e) {
      console.error("Camera mode toggle failed:", e);
    }
  } else {
    if (serverCamElement) {
      serverCamElement.style.display = "none";
      serverCamElement.src = "";
    }
    if (videoElement) videoElement.style.display = "block";

    if (btn) btn.innerText = "Server Direct Cam";
    if (badge) badge.innerText = "Integrated Cam (Browser)";
    if (banner) banner.innerText = "Browser camera stream active.";

    try {
      await fetch("/api/camera/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "browser" })
      });
    } catch (e) {}

    startCamera(preferredDeviceId);
  }
}

let lastCaptureTime = 0;
let isProcessingFrame = false;
let liveTelemetryFrameCounter = 0;

async function captureAndSendLoop(timestamp) {
  if (isCameraPaused || isServerCamActive) {
    requestAnimationFrame(captureAndSendLoop);
    return;
  }

  const now = performance.now();

  // Send frames at smooth ~10 FPS (every 95ms) with strict backpressure protection
  if (now - lastCaptureTime >= 95 && !isProcessingFrame) {
    if (videoElement && (videoElement.readyState >= 2 || videoElement.currentTime > 0)) {
      if (hiddenCtx && hiddenCanvas) {
        lastCaptureTime = now;
        isProcessingFrame = true;

        try {
          hiddenCtx.drawImage(videoElement, 0, 0, 640, 480);
          const dataUrl = hiddenCanvas.toDataURL("image/jpeg", 0.55);

          // Get latest WebAudio metrics from microphone
          const audioPayload = {
            rms: window.currentAudioRms || 0.0,
            is_speech: window.currentIsSpeech || false
          };

          // Ultra-reliable direct HTTP POST pipeline with zero polling bottlenecks
          const response = await fetch("/api/frame/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl, audio: audioPayload })
          });

          if (response.ok) {
            const result = await response.json();
            handleProcessedFrame(result);
          }
        } catch (err) {
          console.warn("Live frame capture notice:", err);
        } finally {
          isProcessingFrame = false;
        }
      }
    }
  }

  requestAnimationFrame(captureAndSendLoop);
}

function toggleCamera() {
  isCameraPaused = !isCameraPaused;
  const btn = document.getElementById("btn-toggle-cam");
  if (btn) {
    btn.innerText = isCameraPaused ? "Resume Cam" : "Pause Cam";
  }
  if (!isCameraPaused && !isServerCamActive) {
    requestAnimationFrame(captureAndSendLoop);
  }
}

// ============================================================================
// 4. DRAWING DETECTIONS & REAL-TIME ENGAGEMENT GAUGE
// ============================================================================
function handleProcessedFrame(data) {
  // Update FPS
  const fpsEl = document.getElementById("fps-val");
  if (fpsEl) fpsEl.innerText = data.fps || "0.0";

  // Update Face counts
  const faceCountEl = document.getElementById("detected-face-count");
  const gazeCountEl = document.getElementById("board-gaze-count");
  const faceList = data.faces || [];
  if (faceCountEl) faceCountEl.innerText = faceList.length;

  let lookingCount = 0;
  if (faceList.length > 0) {
    lookingCount = faceList.filter(f => f.is_looking).length;
  }
  if (gazeCountEl) gazeCountEl.innerText = lookingCount;

  // Draw Bounding Boxes on Overlay Canvas (Faces + Phones)
  drawFaceOverlays(faceList, data.phones || []);

  // Update Circular Engagement Gauge
  if (data.gauge !== undefined) {
    updateEngagementGauge(data.gauge);
  }

  // Update Timetable & Active Period if provided
  if (data.current_period && typeof updateActivePeriodUI === "function") {
    updateActivePeriodUI(data.current_period);
  }

  // Update Audio status
  if (data.audio) {
    const audText = document.getElementById("audio-state-text");
    if (audText) audText.innerText = data.audio.classification || "Quiet";
  }

  // Update Real-Time Breakdown Ratios & Behavior State
  if (data.metrics) {
    const attEl = document.getElementById("metric-attention");
    if (attEl) attEl.innerText = data.metrics.attention + "%";

    const movEl = document.getElementById("metric-movement");
    if (movEl) movEl.innerText = data.metrics.movement + "%";

    const inactEl = document.getElementById("metric-inactivity");
    if (inactEl) inactEl.innerText = data.metrics.inactivity + "s";

    const audRatioEl = document.getElementById("metric-audio");
    if (audRatioEl) audRatioEl.innerText = data.metrics.speech_ratio;

    const bannerEl = document.getElementById("cam-status-msg");
    if (bannerEl && data.metrics.behavior) {
      if (data.metrics.behavior.includes("Phone") || data.metrics.behavior.includes("PHONE")) {
        bannerEl.innerHTML = `<span style="color:#FF9F1C; font-weight:700;">📱 ALERT: Phone usage detected</span> &bull; Student distracted`;
      } else if (data.metrics.behavior.includes("Sleeping")) {
        bannerEl.innerHTML = `<span style="color:#FF5C5C; font-weight:700;">⚠️ ALERT: Sleeping / Head Down detected</span> &bull; Sustained Presence Active`;
      } else if (data.metrics.behavior.includes("Drowsy")) {
        bannerEl.innerHTML = `<span style="color:#F5B83D; font-weight:700;">⚠️ Notice: Inactive / Drowsy posture</span> &bull; Sustained Presence Active`;
      } else {
        bannerEl.innerText = `Observed Behavior: ${data.metrics.behavior} | Sustained Presence Active`;
      }
    }
  }

  // Update Live Attendance Table
  if (data.attendance) {
    updateAttendanceTable(data.attendance);
  }

  // Update Live Telemetry Charts (Dashboard and Analytics) every ~1.2s (every 12 frames)
  liveTelemetryFrameCounter++;
  if (liveTelemetryFrameCounter % 12 === 0) {
    if (typeof window.recordLiveTelemetryPoint === "function") {
      window.recordLiveTelemetryPoint(
        data.gauge,
        data.audio ? data.audio.speech_ratio : 0.20,
        data.is_drop || false,
        data.is_peak || false
      );
    }
  }
}

function drawFaceOverlays(faces, phones = []) {
  if (!overlayCtx || !overlayCanvas) return;

  if (overlayCanvas.clientWidth > 0 && (overlayCanvas.width !== overlayCanvas.clientWidth || overlayCanvas.height !== overlayCanvas.clientHeight)) {
    overlayCanvas.width = overlayCanvas.clientWidth;
    overlayCanvas.height = overlayCanvas.clientHeight;
  }

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const scaleX = overlayCanvas.width / 640.0;
  const scaleY = overlayCanvas.height / 480.0;

  // 1. Draw Detected Phones (Class 67 YOLO)
  if (phones && phones.length > 0) {
    phones.forEach(p => {
      let [px1, py1, px2, py2] = p.bbox;
      const drawPx1 = overlayCanvas.width - (px2 * scaleX);
      const drawPx2 = overlayCanvas.width - (px1 * scaleX);
      const drawPy1 = py1 * scaleY;
      const drawPy2 = py2 * scaleY;
      const phoneW = drawPx2 - drawPx1;
      const phoneH = drawPy2 - drawPy1;

      // Draw dashed amber phone border
      overlayCtx.strokeStyle = "#FF9F1C";
      overlayCtx.lineWidth = 2.0;
      overlayCtx.setLineDash([5, 4]);
      overlayCtx.strokeRect(drawPx1, drawPy1, phoneW, phoneH);
      overlayCtx.setLineDash([]);

      // Phone chip
      overlayCtx.fillStyle = "rgba(255, 159, 28, 0.95)";
      overlayCtx.beginPath();
      overlayCtx.roundRect(drawPx1, Math.max(0, drawPy1 - 18), 75, 18, 4);
      overlayCtx.fill();
      overlayCtx.fillStyle = "#0A0A0A";
      overlayCtx.font = "bold 11px Inter, sans-serif";
      overlayCtx.fillText("📱 Phone", drawPx1 + 6, Math.max(0, drawPy1 - 5));
    });
  }

  // 2. Draw Detected Faces
  faces.forEach(face => {
    let [x1, y1, x2, y2] = face.bbox;
    // Mirrored display calculation
    const origX1 = x1 * scaleX;
    const origX2 = x2 * scaleX;
    const drawX1 = overlayCanvas.width - origX2;
    const drawX2 = overlayCanvas.width - origX1;
    const drawY1 = y1 * scaleY;
    const drawY2 = y2 * scaleY;
    const boxW = drawX2 - drawX1;
    const boxH = drawY2 - drawY1;

    // Determine state color & label
    const behavior = face.behavior || (face.is_looking ? "Focused" : "Away");
    let boxColor = "#C5F82A";
    let dotColor = "#C5F82A";
    let statusText = behavior;

    if (face.has_phone || behavior.includes("Phone") || behavior.includes("PHONE")) {
      boxColor = "#FF9F1C"; // Amber/Orange phone alert
      dotColor = "#FF9F1C";
      statusText = "📱 PHONE USAGE / DISTRACTED";
    } else if (behavior.includes("Sleeping")) {
      boxColor = "#FF5C5C";
      dotColor = "#FF5C5C";
      statusText = "⚠️ SLEEPING / HEAD DOWN";
    } else if (behavior.includes("Drowsy")) {
      boxColor = "#F5B83D";
      dotColor = "#F5B83D";
      statusText = "Drowsy / Inactive";
    } else if (behavior.includes("Down")) {
      boxColor = "#F5B83D";
      dotColor = "#F5B83D";
      statusText = "Looking Down";
    } else if (behavior.includes("Away")) {
      boxColor = "#F5B83D";
      dotColor = "#F5B83D";
      statusText = "Looking Away";
    }

    // Draw Dynamic Bounding Box
    overlayCtx.strokeStyle = boxColor;
    overlayCtx.lineWidth = 2.5;
    overlayCtx.strokeRect(drawX1, drawY1, boxW, boxH);

    // Corner accents
    const cornerSize = 12;
    overlayCtx.strokeStyle = "#FFFFFF";
    overlayCtx.lineWidth = 3.5;
    // Top-left
    overlayCtx.beginPath();
    overlayCtx.moveTo(drawX1, drawY1 + cornerSize);
    overlayCtx.lineTo(drawX1, drawY1);
    overlayCtx.lineTo(drawX1 + cornerSize, drawY1);
    overlayCtx.stroke();
    // Top-right
    overlayCtx.beginPath();
    overlayCtx.moveTo(drawX2 - cornerSize, drawY1);
    overlayCtx.lineTo(drawX2, drawY1);
    overlayCtx.lineTo(drawX2, drawY1 + cornerSize);
    overlayCtx.stroke();

    // Dark Pill Label with status dot
    let displayName = face.name;
    if (isAnonymousMode && displayName && displayName !== "Unknown") {
      displayName = `Student #${face.track_id || 1}`;
    }
    const labelText = `${displayName} (${statusText})`;
    overlayCtx.font = "bold 12px Inter, sans-serif";
    const textW = overlayCtx.measureText(labelText).width;
    const pillW = textW + 28;
    const pillH = 24;
    const pillX = drawX1;
    const pillY = Math.max(0, drawY1 - pillH - 6);

    // Pill background
    overlayCtx.fillStyle = "rgba(10, 10, 10, 0.85)";
    overlayCtx.beginPath();
    overlayCtx.roundRect(pillX, pillY, pillW, pillH, 12);
    overlayCtx.fill();
    overlayCtx.strokeStyle = boxColor;
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();

    // Focus / Sleeping indicator dot
    overlayCtx.fillStyle = dotColor;
    overlayCtx.beginPath();
    overlayCtx.arc(pillX + 10, pillY + pillH / 2, 4, 0, Math.PI * 2);
    overlayCtx.fill();

    // Student Name text
    overlayCtx.fillStyle = "#FFFFFF";
    overlayCtx.fillText(labelText, pillX + 20, pillY + 16);
  });
}

function updateEngagementGauge(score) {
  const scoreEl = document.getElementById("gauge-score");
  const circleEl = document.getElementById("gauge-circle");
  if (!scoreEl || !circleEl) return;

  const rounded = Math.round(score);
  scoreEl.innerText = rounded + "%";

  const offset = GAUGE_CIRCUMFERENCE * (1 - Math.max(0, Math.min(100, score)) / 100.0);
  circleEl.style.strokeDashoffset = offset;

  if (score < 45) {
    circleEl.style.stroke = "#FF5C5C";
  } else if (score < 65) {
    circleEl.style.stroke = "#F5B83D";
  } else {
    circleEl.style.stroke = "#C5F82A";
  }

  // Update Stat card and Marquee ticker
  const statAvg = document.getElementById("stat-engagement-avg");
  if (statAvg) statAvg.innerText = rounded + "%";
  const tickEl = document.getElementById("ticker-eng-val");
  if (tickEl) tickEl.innerText = rounded + "%";
}

function updateAttendanceTable(records) {
  const liveTbody = document.getElementById("live-attendance-tbody");
  const fullTbody = document.getElementById("full-attendance-tbody");
  if (!records || records.length === 0) return;

  let presentCount = 0;
  let rowsHtml = "";

  records.forEach(r => {
    let chipClass = "absent";
    if (r.status === "Present") {
      chipClass = "present";
      presentCount++;
    } else if (r.status === "Late") {
      chipClass = "late";
      presentCount++;
    } else if (r.status === "Partial") {
      chipClass = "partial";
    } else if (r.status.includes("Ready")) {
      chipClass = "ready";
    }

    // Dynamic verified status with percentage feedback
    const sRatio = r.sustained_ratio !== undefined ? Math.round(r.sustained_ratio) : (r.sustained_verified ? 100 : 0);
    let verifiedText = '<span style="color:var(--text-dim);">Unverified (0%)</span>';
    if (r.sustained_verified || r.status === "Present" || r.status === "Late") {
      verifiedText = `<span style="color:var(--accent); font-weight:600;">✓ Verified (${Math.max(70, sRatio)}%)</span>`;
    } else if (r.status.includes("Ready")) {
      verifiedText = `<span style="color:var(--accent); font-weight:600;">👀 In Camera (Click Start Session to Record)</span>`;
    } else if (r.status === "Standby") {
      verifiedText = `<span style="color:var(--text-dim);">Standby (No Active Session)</span>`;
    } else if (sRatio > 0 || r.status === "Partial") {
      verifiedText = `<span style="color:#F5B83D; font-weight:600;">⚡ Evaluating (${sRatio}%)</span>`;
    }

    let displayName = r.student_name;
    let displayRoll = r.roll_number;
    if (isAnonymousMode && displayName && displayName !== "Unknown") {
      displayName = `Student #${r.student_id || 1}`;
      displayRoll = `Roll #${String(r.student_id || 1).padStart(3, '0')}`;
    }

    // Dynamic presence time formatting (seconds vs minutes)
    const sec = Number(r.total_presence_seconds) || 0.0;
    let presenceDisplay = `${Math.round(sec)}s`;
    if (sec >= 60) {
      const m = Math.floor(sec / 60);
      const s = Math.round(sec % 60);
      presenceDisplay = `${m}m ${s}s`;
    }

    rowsHtml += `
      <tr>
        <td><strong>${displayRoll}</strong></td>
        <td>${displayName}</td>
        <td><span class="status-chip ${chipClass}">● ${r.status}</span></td>
        <td>${verifiedText}</td>
        <td>${r.first_seen}</td>
        <td>${r.last_seen}</td>
        <td>${presenceDisplay}</td>
      </tr>
    `;
  });

  if (liveTbody) liveTbody.innerHTML = rowsHtml;
  if (fullTbody) fullTbody.innerHTML = rowsHtml;

  const statPres = document.getElementById("stat-present-count");
  if (statPres) statPres.innerText = presentCount;

  const tickerPres = document.getElementById("ticker-present-count");
  if (tickerPres) tickerPres.innerText = `${presentCount} PRESENT`;
}

// ============================================================================
// 5. DYNAMIC DATA FETCHING ACROSS SPA VIEWS
// ============================================================================

// --- ENROLLMENT DIRECTORY ---
async function fetchEnrolledStudents() {
  const tbody = document.getElementById("students-tbody");
  const countEl = document.getElementById("enrolled-count");
  if (!tbody) return;

  try {
    const res = await fetch("/api/students");
    const data = await res.json();
    if (data.success) {
      const students = data.students || [];
      if (countEl) countEl.innerText = students.length;

      if (students.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 40px;">
              No students enrolled yet. Click "+ New Student Enrollment" to register biometric embeddings.
            </td>
          </tr>
        `;
        return;
      }

      let html = "";
      students.forEach(s => {
        let displayName = isAnonymousMode ? `Student #${s.id}` : s.name;
        let displayRoll = isAnonymousMode ? `Roll #${String(s.id).padStart(3, '0')}` : s.roll_number;
        html += `
          <tr>
            <td><strong>#${s.id}</strong></td>
            <td>${displayRoll}</td>
            <td style="font-weight: 600; color: #FFF;">${displayName}</td>
            <td><span class="badge badge-lime">5 / 5 Averaged</span></td>
            <td>${s.created_at || "Just now"}</td>
            <td>
              <button class="btn btn-secondary btn-sm" onclick="openEditModal(${s.id}, '${s.name}', '${s.roll_number}')">Edit</button>
              <button class="btn btn-danger btn-sm" onclick="deleteStudent(${s.id}, '${s.name}')">Delete</button>
            </td>
          </tr>
        `;
      });
      tbody.innerHTML = html;
    }
  } catch (err) {
    console.error("Error fetching students:", err);
  }
}

// --- ATTENDANCE REGISTER ---
async function fetchFullAttendance() {
  const tbody = document.getElementById("full-attendance-tbody");
  const subText = document.getElementById("attendance-roster-sub");
  if (!tbody) return;

  try {
    const res = await fetch("/api/attendance/current");
    const data = await res.json();
    if (data.success) {
      const active = data.active_session;
      if (subText) {
        subText.innerHTML = active 
          ? `Active Session: <strong>${active.name}</strong> (${active.subject}) &bull; ${active.present_count} Present`
          : "Session archive register &bull; Sustained presence log";
      }

      const records = data.records || [];
      if (records.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 40px;">
              No attendance records available. Start a session from the Live Session dashboard to begin tracking.
            </td>
          </tr>
        `;
        return;
      }

      let html = "";
      records.forEach(r => {
        let chipClass = "absent";
        if (r.status === "Present") chipClass = "present";
        else if (r.status === "Late") chipClass = "late";
        else if (r.status === "Partial") chipClass = "partial";

        let displayName = isAnonymousMode && r.student_name !== "Unknown" ? `Student #${r.student_id || 1}` : r.student_name;
        let displayRoll = isAnonymousMode && r.student_name !== "Unknown" ? `Roll #${String(r.student_id || 1).padStart(3, '0')}` : r.roll_number;

        const sec = Number(r.total_presence_seconds) || 0.0;
        let presenceDisplay = `${Math.round(sec)}s`;
        if (sec >= 60) {
          const m = Math.floor(sec / 60);
          const s = Math.round(sec % 60);
          presenceDisplay = `${m}m ${s}s`;
        }

        html += `
          <tr>
            <td><strong>${displayRoll}</strong></td>
            <td>${displayName}</td>
            <td><span class="status-chip ${chipClass}">● ${r.status}</span></td>
            <td>${r.sustained_verified ? '<span style="color:var(--accent); font-weight:600;">✓ Verified (&ge;70%)</span>' : '<span style="color:var(--text-dim);">Unverified</span>'}</td>
            <td>${r.first_seen}</td>
            <td>${r.last_seen}</td>
            <td>${presenceDisplay}</td>
          </tr>
        `;
      });
      tbody.innerHTML = html;
    }
  } catch (err) {
    console.error("Error fetching attendance:", err);
  }
}

// --- SESSION HISTORY ARCHIVE ---
async function fetchSessionHistory() {
  const container = document.getElementById("history-cards-container");
  if (!container) return;

  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    if (data.success) {
      const sessions = data.sessions || [];
      if (sessions.length === 0) {
        container.innerHTML = `
          <div style="grid-column: 1 / -1; background: var(--bg-card); border: var(--border-subtle); border-radius: var(--radius-card); padding: 50px; text-align: center; color: var(--text-muted);">
            No past sessions recorded yet. Completed lectures will appear here automatically.
          </div>
        `;
        return;
      }

      let html = "";
      sessions.forEach(s => {
        html += `
          <div class="feature-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
              <span class="pill-badge" style="margin: 0;"><span class="pill-dot"></span> Session #${s.id}</span>
              <span style="font-size: 12px; color: var(--text-muted);">${s.started_at}</span>
            </div>
            <h3 class="feature-title">${s.name}</h3>
            <p class="feature-desc" style="margin-bottom: 16px;">
              Subject: <strong>${s.subject}</strong><br>
              Duration: <strong>${Math.floor(s.duration_seconds / 60)} min</strong> &bull; Present: <strong style="color: var(--accent);">${s.present_count} / ${s.total_enrolled}</strong><br>
              Average Engagement: <strong style="color: var(--accent);">${s.avg_engagement}%</strong>
            </p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
              <a href="/reports/session/${s.id}/csv" class="btn btn-secondary btn-sm" download>CSV Export &darr;</a>
              <a href="/reports/session/${s.id}/html" target="_blank" class="btn btn-primary btn-sm">HTML Report &rarr;</a>
            </div>
          </div>
        `;
      });
      container.innerHTML = html;
    }
  } catch (err) {
    console.error("Error fetching history:", err);
  }
}

// --- INSIGHTS TIMELINE ---
async function fetchInsightsTimeline() {
  const container = document.getElementById("full-insights-container");
  if (!container) return;

  try {
    const res = await fetch("/api/insights");
    const data = await res.json();
    if (data.success) {
      const suggestions = data.suggestions || [];
      if (suggestions.length === 0) {
        container.innerHTML = `
          <div class="suggestion-card">
            <div class="suggestion-meta">
              <span class="suggestion-badge">Assistant Ready</span>
              <span class="suggestion-time">Active</span>
            </div>
            <div class="suggestion-text">
              No attention drops recorded yet. Insights will appear dynamically as class intervals unfold.
            </div>
          </div>
        `;
        return;
      }

      let html = "";
      suggestions.forEach(s => {
        const dropLabel = s.drop_percentage > 0 ? `<span style="color:#FF5C5C; font-size:12px; font-weight:700;">-${s.drop_percentage}% Drop</span>` : "";
        html += `
          <div class="suggestion-card">
            <div class="suggestion-meta">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span class="suggestion-badge">${s.trigger_type.toUpperCase()}</span>
                ${dropLabel}
              </div>
              <span class="suggestion-time">${s.time} &bull; ${s.source} (${s.model_name})</span>
            </div>
            <div class="suggestion-text">${s.suggestion}</div>
          </div>
        `;
      });
      container.innerHTML = html;
    }
  } catch (err) {
    console.error("Error fetching insights:", err);
  }
}

// ============================================================================
// 6. SUGGESTIONS & FEEDBACK
// ============================================================================
function displayNewSuggestion(sugg) {
  const container = document.getElementById("live-suggestions-container");
  if (!container) return;

  const card = document.createElement("div");
  card.className = "suggestion-card";
  const dropLabel = sugg.drop_percentage > 0 ? `<span style="color:#FF5C5C; font-size:12px; font-weight:700;">-${sugg.drop_percentage}% Drop</span>` : "";

  card.innerHTML = `
    <div class="suggestion-meta">
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="suggestion-badge">${sugg.trigger_type.toUpperCase()}</span>
        ${dropLabel}
      </div>
      <span class="suggestion-time">${sugg.time} &bull; ${sugg.source} (${sugg.model_name})</span>
    </div>
    <div class="suggestion-text">${sugg.suggestion}</div>
  `;

  container.prepend(card);
  showToast(`💡 New AI Teaching Insight: ${sugg.trigger_type}`);
}

// ============================================================================
// 7. SESSION MODAL & LIFECYCLE
// ============================================================================
function toggleSessionModal() {
  const modal = document.getElementById("session-modal");
  const navBtn = document.getElementById("nav-session-btn");
  if (!modal) return;

  if (navBtn && navBtn.innerText.includes("Stop")) {
    if (confirm("Are you sure you want to stop the active session?")) {
      stopSession();
    }
  } else {
    modal.style.display = "flex";
  }
}

function closeSessionModal() {
  const modal = document.getElementById("session-modal");
  if (modal) modal.style.display = "none";
}

async function submitStartSession() {
  let name = document.getElementById("modal-session-name").value.trim();
  let subject = document.getElementById("modal-session-subject").value.trim();

  const autoBind = document.getElementById("session-auto-timetable");
  if (autoBind && autoBind.checked && window.currentPeriodData) {
    if (!name) name = `${window.currentPeriodData.subject_code} Period ${window.currentPeriodData.period} Lecture`;
    if (!subject) subject = window.currentPeriodData.subject_title;
  }

  if (!name) name = "LNCT B.Tech Lecture Session";
  if (!subject) subject = "CSE (AIML)";

  try {
    const res = await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, subject })
    });
    const data = await res.json();
    if (data.success) {
      closeSessionModal();
      showToast("🚀 Session started successfully! Tracking active period attention.");
      if (typeof window.resetChartsForNewSession === "function") {
        window.resetChartsForNewSession();
      }
      handleSessionStateChange({ active: true, session: data.session });
      // If server cam wasn't active, ensure camera stream is running
      if (!isStreaming && !isServerCamActive) {
        startCamera(preferredDeviceId);
      }
    }
  } catch (err) {
    alert("Error starting session: " + err.message);
  }
}

async function stopSession() {
  // Immediately update UI optimistically so there is zero lag
  handleSessionStateChange({ active: false, session: null });
  closeSessionModal();
  showToast("⏹ Session ended. Final report compiled.");

  try {
    const res = await fetch("/api/session/stop", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      console.log("Session stopped successfully:", data);
    }
  } catch (err) {
    console.warn("Session stop background notice:", err);
  }
}

function handleSessionStateChange(data) {
  const navBtn = document.getElementById("nav-session-btn");
  const heroBtn = document.getElementById("btn-hero-session");
  const badgeState = document.getElementById("badge-session-state");
  const tickerStatus = document.getElementById("ticker-session-status");

  if (data.active) {
    if (navBtn) navBtn.innerText = "⏹ Stop Session";
    if (heroBtn) heroBtn.innerText = "⏹ End Current Session";
    const sName = (data.session && data.session.name) ? data.session.name : "Classroom Session";
    if (badgeState) badgeState.innerText = `Active: ${sName}`;
    if (tickerStatus) tickerStatus.innerText = `LIVE SESSION: ${sName}`;
    startSessionTimer();
  } else {
    if (navBtn) navBtn.innerText = "▶ Start Session";
    if (heroBtn) heroBtn.innerText = "▶ Start Live Session";
    if (badgeState) badgeState.innerText = "Session Ended";
    if (tickerStatus) tickerStatus.innerText = "NO ACTIVE SESSION";
    stopSessionTimer();
  }
}

function startSessionTimer() {
  sessionStartTimestamp = Date.now();
  if (sessionTimerInterval) clearInterval(sessionTimerInterval);
  sessionTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTimestamp) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const secs = String(elapsed % 60).padStart(2, "0");
    const el = document.getElementById("stat-session-timer");
    if (el) el.innerText = `${mins}:${secs}`;
  }, 1000);
}

function stopSessionTimer() {
  if (sessionTimerInterval) clearInterval(sessionTimerInterval);
  sessionTimerInterval = null;
}

// ============================================================================
// 8. REPORT EXPORTS & DOWNLOAD ACTIONS
// ============================================================================
async function downloadActiveCSV() {
  try {
    const res = await fetch("/api/session/status");
    const data = await res.json();
    if (data.active && data.session) {
      window.location.href = `/reports/session/${data.session.id}/csv`;
    } else {
      showToast("ℹ️ No active session. Showing most recent session CSV.");
      window.location.href = `/reports/session/1/csv`;
    }
  } catch (err) {
    alert("Export error: " + err.message);
  }
}

async function openPrintableReport() {
  try {
    const res = await fetch("/api/session/status");
    const data = await res.json();
    const sessionId = data.active && data.session ? data.session.id : 1;
    window.open(`/reports/session/${sessionId}/html`, "_blank");
  } catch (err) {
    alert("Report error: " + err.message);
  }
}

// ============================================================================
// 9. ANONYMOUS & PRIVACY STATE
// ============================================================================
function applyAnonymousMode(isAnon) {
  isAnonymousMode = Boolean(isAnon);
  const lbl = document.getElementById("anon-label");
  const btn = document.getElementById("btn-anon-toggle");
  if (lbl) {
    lbl.innerText = isAnonymousMode ? "🔒 Anon: ON" : "👁️ Anon: OFF";
  }
  if (btn) {
    if (isAnonymousMode) {
      btn.style.borderColor = "var(--accent)";
      btn.style.color = "var(--accent)";
      btn.style.backgroundColor = "rgba(197, 248, 42, 0.15)";
    } else {
      btn.style.borderColor = "var(--border-subtle)";
      btn.style.color = "var(--text-main)";
      btn.style.backgroundColor = "transparent";
    }
  }

  // Refresh current view data dynamically
  const activeNav = document.querySelector(".nav-link.active");
  const currentView = activeNav ? activeNav.id.replace("nav-item-", "") : "live";
  if (currentView === "enrollment") {
    fetchEnrolledStudents();
  } else if (currentView === "attendance") {
    fetchFullAttendance();
  } else if (currentView === "live") {
    fetchFullAttendance();
  }
}

async function toggleAnonymousMode() {
  try {
    const nextState = !isAnonymousMode;
    const res = await fetch("/api/privacy/anonymous-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: nextState })
    });
    const data = await res.json();
    if (data.success) {
      applyAnonymousMode(data.anonymous_mode);
      showToast(data.anonymous_mode ? "🔒 Anonymous Mode Activated (Names Masked)" : "👁️ Anonymous Mode Deactivated");
    }
  } catch (err) {
    console.error("Failed to toggle anonymous mode:", err);
  }
}

async function initAnonymousState() {
  try {
    const res = await fetch("/api/system/status");
    const data = await res.json();
    if (data && data.anonymous_mode !== undefined) {
      applyAnonymousMode(data.anonymous_mode);
    }
  } catch (e) {
    console.warn("Could not initialize anonymous state:", e);
  }
}

function initConsentState() {
  if (localStorage.getItem("offline_consent_dismissed") === "true") {
    const banner = document.getElementById("consent-banner");
    if (banner) banner.style.display = "none";
  }
}

function dismissConsent() {
  localStorage.setItem("offline_consent_dismissed", "true");
  const banner = document.getElementById("consent-banner");
  if (banner) banner.style.display = "none";
}

// ============================================================================
// 10. EDIT STUDENT MODAL & DELETION
// ============================================================================
function openEditModal(studentId, currentName, currentRoll) {
  const modal = document.getElementById("edit-modal");
  if (!modal) return;
  document.getElementById("edit-student-id").value = studentId;
  document.getElementById("edit-student-name").value = currentName;
  document.getElementById("edit-student-roll").value = currentRoll;
  modal.style.display = "flex";
}

function closeEditModal() {
  const modal = document.getElementById("edit-modal");
  if (modal) modal.style.display = "none";
}

async function saveStudentEdit() {
  const id = document.getElementById("edit-student-id").value;
  const name = document.getElementById("edit-student-name").value.trim();
  const roll_number = document.getElementById("edit-student-roll").value.trim();

  if (!name || !roll_number) {
    alert("Please provide both name and roll number.");
    return;
  }

  try {
    const res = await fetch(`/api/students/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, roll_number })
    });
    const data = await res.json();
    if (data.success) {
      closeEditModal();
      showToast(`Student #${id} updated successfully.`);
      fetchEnrolledStudents();
    }
  } catch (err) {
    alert("Error updating student: " + err.message);
  }
}

async function deleteStudent(studentId, name) {
  if (!confirm(`Are you sure you want to delete ${name} (ID: ${studentId})?`)) return;
  try {
    const res = await fetch(`/api/students/${studentId}`, { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      showToast(`Student ${name} deleted.`);
      fetchEnrolledStudents();
    }
  } catch (err) {
    alert("Error deleting student: " + err.message);
  }
}

async function confirmDeleteAll() {
  if (!confirm("⚠️ CAUTION: Delete ALL enrolled students and biometric embeddings? This action cannot be undone.")) return;
  try {
    const res = await fetch("/api/privacy/delete-all", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      showToast("All biometric data purged successfully.");
      fetchEnrolledStudents();
    }
  } catch (err) {
    alert("Error purging data: " + err.message);
  }
}

// ============================================================================
// 11. TOAST NOTIFICATIONS
// ============================================================================
function showToast(message) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<span>●</span> <span>${message}</span>`;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ============================================================================
// 12. BROWSER WEBAUDIO REAL-TIME MICROPHONE STREAMING
// ============================================================================
let browserAudioContext = null;
let browserAnalyser = null;
let browserAudioInterval = null;

async function initBrowserAudio() {
  try {
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    browserAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = browserAudioContext.createMediaStreamSource(micStream);
    browserAnalyser = browserAudioContext.createAnalyser();
    browserAnalyser.fftSize = 512;
    source.connect(browserAnalyser);

    const unlockAudio = () => {
      if (browserAudioContext && browserAudioContext.state === "suspended") {
        browserAudioContext.resume().catch(() => {});
      }
    };
    document.addEventListener("click", unlockAudio, { passive: true });
    document.addEventListener("keydown", unlockAudio, { passive: true });

    const dataArray = new Float32Array(browserAnalyser.fftSize);

    if (browserAudioInterval) clearInterval(browserAudioInterval);
    browserAudioInterval = setInterval(() => {
      if (!browserAnalyser) return;
      if (browserAudioContext && browserAudioContext.state === "suspended") {
        browserAudioContext.resume().catch(() => {});
      }
      browserAnalyser.getFloatTimeDomainData(dataArray);
      let sumSq = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sumSq += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sumSq / dataArray.length);
      const isSpeech = rms > 0.003;
      window.currentAudioRms = rms;
      window.currentIsSpeech = isSpeech;

      if (socket && socket.connected) {
        socket.emit("client_audio_metric", { rms: rms, is_speech: isSpeech });
      }
    }, 150);

    console.log("🎤 Browser microphone connected with WebAudio AGC.");
  } catch (e) {
    console.log("Browser microphone optional: using host sounddevice capture.");
  }
}
