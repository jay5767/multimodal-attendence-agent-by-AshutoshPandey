/**
 * LNCT Timetable & Real-Time Period Attention Tracker
 * Lakshmi Narain College of Technology, Bhopal
 * Class: B.Tech Vth Semester | CSE (AIML) - Section B | Room 4-B9 (CV Raman Block)
 */

let timetableData = null;
let currentPeriodData = null;
let clockInterval = null;
let simulationDay = null;
let simulationTime = null;

document.addEventListener("DOMContentLoaded", () => {
  initRealTimeClock();
  fetchTimetableData();
});

/**
 * 1. REAL-TIME DIGITAL CLOCK (Ticking every 1000ms)
 */
function initRealTimeClock() {
  updateClockDisplay();
  if (clockInterval) clearInterval(clockInterval);
  clockInterval = setInterval(updateClockDisplay, 1000);
}

function updateClockDisplay() {
  const now = new Date();
  
  // Day name & date formatted
  const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  
  const dayName = days[now.getDay()];
  const dayShort = dayName.substring(0, 3);
  const dateStr = `${now.getDate()} ${months[now.getMonth()]} ${now.getFullYear()}`;
  
  // 12-hour time format with seconds
  let hours = now.getHours();
  const minutes = String(now.getMinutes()).padStart(2, "0");
  const seconds = String(now.getSeconds()).padStart(2, "0");
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12;
  hours = hours ? hours : 12; // 0 becomes 12
  const hoursStr = String(hours).padStart(2, "0");
  const timeStr = `${hoursStr}:${minutes}:${seconds} ${ampm}`;

  // 1. Update Navbar Digital Clock Badge
  const navDayDate = document.getElementById("clock-day-date");
  const navTime = document.getElementById("clock-time");
  if (navDayDate) navDayDate.innerText = `${dayShort}, ${now.getDate()} ${months[now.getMonth()]}`;
  if (navTime) navTime.innerText = timeStr;

  // 2. Update Timetable View Prominent Clock
  const ttDay = document.getElementById("tt-clock-day");
  const ttDate = document.getElementById("tt-clock-date");
  const ttTime = document.getElementById("tt-clock-time");
  if (ttDay) ttDay.innerText = dayName;
  if (ttDate) ttDate.innerText = dateStr;
  if (ttTime) ttTime.innerText = timeStr;

  // 3. Update Period Countdown Timer if active
  updatePeriodCountdown(now);
}

function updatePeriodCountdown(now) {
  if (!currentPeriodData || !currentPeriodData.is_active) return;
  const remEl = document.getElementById("tt-active-remaining");
  if (!remEl) return;

  const currentMin = now.getHours() * 60 + now.getMinutes();
  const [endH, endM] = currentPeriodData.time_slot.split("-")[1].trim().split(":").map(Number);
  const endTotalMin = endH * 60 + endM;

  const diffMins = endTotalMin - currentMin;
  const diffSecs = 60 - now.getSeconds();

  if (diffMins > 0) {
    remEl.innerText = `${diffMins - 1}m ${diffSecs}s remaining`;
  } else {
    // Period boundary reached, re-fetch
    fetchCurrentPeriod();
  }
}

/**
 * 2. FETCH TIMETABLE DATA & ACTIVE PERIOD
 */
async function fetchTimetableData() {
  try {
    const res = await fetch("/api/timetable/all");
    const json = await res.json();
    if (json.success) {
      timetableData = json.timetable;
      renderWeeklyGrid(timetableData);
      renderSubjectCatalog(timetableData);
      renderSubjectAttentionStats(timetableData.subject_stats);
    }
  } catch (err) {
    console.error("Could not fetch timetable all:", err);
  }

  fetchCurrentPeriod();
}

async function fetchCurrentPeriod() {
  try {
    let url = "/api/timetable/current";
    const params = [];
    if (simulationDay) {
      params.push(`day=${simulationDay}`);
    } else {
      const now = new Date();
      const days = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
      params.push(`day=${days[now.getDay()]}`);
    }
    if (simulationTime) {
      params.push(`time=${simulationTime}`);
    } else {
      const now = new Date();
      const hh = String(now.getHours()).padStart(2, '0');
      const mm = String(now.getMinutes()).padStart(2, '0');
      params.push(`time=${hh}:${mm}`);
    }
    if (params.length > 0) url += `?${params.join("&")}`;

    const res = await fetch(url);
    const json = await res.json();
    if (json.success && json.period) {
      currentPeriodData = json.period;
      updateActivePeriodUI(json.period);
      highlightActiveGridCell(json.period);
    }
  } catch (err) {
    console.error("Could not fetch current period:", err);
  }
}

/**
 * 3. UPDATE ACTIVE PERIOD UI
 */
function updateActivePeriodUI(p) {
  const badgeState = document.getElementById("tt-active-badge");
  const subCode = document.getElementById("tt-active-subcode");
  const subTitle = document.getElementById("tt-active-title");
  const faculty = document.getElementById("tt-active-faculty");
  const contact = document.getElementById("tt-active-contact");
  const room = document.getElementById("tt-active-room");
  const slot = document.getElementById("tt-active-slot");
  const remaining = document.getElementById("tt-active-remaining");

  if (badgeState) {
    badgeState.innerText = p.is_active ? `● LIVE: PERIOD ${p.period}` : "● STANDBY: OFF-HOURS";
    badgeState.style.color = p.is_active ? "var(--accent)" : "#9CA3A0";
  }

  if (subCode) subCode.innerText = p.subject_code;
  if (subTitle) subTitle.innerText = p.subject_title;
  if (faculty) faculty.innerText = p.faculty;
  if (contact) {
    const teachingStr = p.faculty_subject || p.faculty_contact || p.subject_title;
    contact.innerText = teachingStr.startsWith("📚") ? teachingStr : `📚 Teaching: ${teachingStr}`;
  }
  if (room) room.innerText = p.room;
  if (slot) slot.innerText = p.time_slot;
  if (remaining) remaining.innerText = p.time_remaining;

  // Update "In this subject, how many students are paying attention" section
  const attRateEl = document.getElementById("tt-subject-att-rate");
  const attBarEl = document.getElementById("tt-subject-att-bar");
  const phoneDistractEl = document.getElementById("tt-subject-phone");
  const sleepDistractEl = document.getElementById("tt-subject-sleep");
  const awayDistractEl = document.getElementById("tt-subject-away");
  const payingCountEl = document.getElementById("tt-paying-count");
  const totalInClassEl = document.getElementById("tt-total-in-class");

  const rate = p.attention_rate || 84.5;
  if (attRateEl) attRateEl.innerText = rate + "%";
  if (attBarEl) attBarEl.style.width = rate + "%";

  // Calculate live head count based on current session or baseline
  const enrolledCount = parseInt(document.getElementById("stat-present-count")?.innerText || "24", 10) || 24;
  const payingCount = Math.round(enrolledCount * (rate / 100.0));
  if (payingCountEl) payingCountEl.innerText = payingCount;
  if (totalInClassEl) totalInClassEl.innerText = enrolledCount;

  if (phoneDistractEl) phoneDistractEl.innerText = p.distracted_phone || 0;
  if (sleepDistractEl) sleepDistractEl.innerText = p.distracted_sleep || 0;
  if (awayDistractEl) awayDistractEl.innerText = p.distracted_away || 0;

  // Update Start Session Modal link text if open
  const modalPeriodName = document.getElementById("modal-current-period-name");
  if (modalPeriodName) {
    modalPeriodName.innerText = `${p.subject_code} (${p.subject_title.split("(")[0].trim()})`;
  }
}

/**
 * 4. RENDER WEEKLY TIMETABLE GRID
 */
function renderWeeklyGrid(data) {
  const tbody = document.getElementById("tt-grid-body");
  if (!tbody || !data || !data.weekly_schedule) return;

  tbody.innerHTML = "";
  const days = ["MON", "TUE", "WED", "THU", "FRI", "SAT"];

  days.forEach(day => {
    const slots = data.weekly_schedule[day] || [];
    const tr = document.createElement("tr");
    tr.id = `tt-row-${day}`;
    tr.style.borderBottom = "1px solid rgba(255,255,255,0.06)";

    // Day header column
    let html = `
      <td style="padding:14px 16px; font-weight:700; color:var(--accent); font-family:var(--font-heading); background:rgba(255,255,255,0.02); text-align:center;">
        ${day}
      </td>
    `;

    // 9 period columns
    slots.forEach((s, idx) => {
      const isLunch = (s.code === "LUNCH");
      const subStyle = isLunch
        ? "background:rgba(255,255,255,0.03); color:var(--text-muted); font-size:11px; text-align:center;"
        : "background:transparent; text-align:center; font-size:12px; cursor:pointer;";
      
      const cellId = `tt-cell-${day}-${s.period}`;
      html += `
        <td id="${cellId}" style="padding:10px 8px; ${subStyle}; border-left:1px solid rgba(255,255,255,0.04);" onclick="selectTimetableCell('${day}', '${s.period}', '${s.code}')">
          <div style="font-weight:700; color:${isLunch ? '#9CA3A0' : '#FFFFFF'}; margin-bottom:2px;">${s.code}</div>
          <div style="font-size:10px; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:85px; margin:0 auto;">${s.faculty}</div>
        </td>
      `;
    });

    tr.innerHTML = html;
    tbody.appendChild(tr);
  });
}

function highlightActiveGridCell(period) {
  // Remove existing active highlights
  document.querySelectorAll(".tt-cell-highlight").forEach(el => {
    el.classList.remove("tt-cell-highlight");
    el.style.border = "none";
    el.style.backgroundColor = "transparent";
  });

  if (!period || !period.day || !period.period) return;
  const cell = document.getElementById(`tt-cell-${period.day}-${period.period}`);
  if (cell) {
    cell.classList.add("tt-cell-highlight");
    cell.style.backgroundColor = "rgba(197, 248, 42, 0.15)";
    cell.style.outline = "2px solid var(--accent)";
    cell.style.borderRadius = "6px";
  }
}

/**
 * 5. INTERACTIVE PERIOD SIMULATION
 */
function simulatePeriod(day, timeStr) {
  simulationDay = day;
  simulationTime = timeStr;
  showToast(`Simulating ${day} at ${timeStr}...`);
  fetchCurrentPeriod();
}

function resetSimulation() {
  simulationDay = null;
  simulationTime = null;
  showToast("Reverted to Real-Time Clock & Timetable tracking.");
  fetchCurrentPeriod();
}

function selectTimetableCell(day, periodNum, subjectCode) {
  if (subjectCode === "LUNCH") return;
  // Map period to typical start time
  const timeMap = {
    "I": "10:45",
    "II": "11:30",
    "III": "12:15",
    "IV": "13:00",
    "VI": "14:20",
    "VII": "15:00",
    "VIII": "15:45",
    "IX": "16:30"
  };
  simulatePeriod(day, timeMap[periodNum] || "11:30");
}

/**
 * 6. SUBJECT ATTENTION ANALYTICS & BREAKDOWN
 */
function renderSubjectAttentionStats(stats) {
  const container = document.getElementById("tt-subject-stats-grid");
  if (!container || !stats) return;

  container.innerHTML = "";
  Object.keys(stats).forEach(code => {
    const s = stats[code];
    const card = document.createElement("div");
    card.style.background = "var(--bg-card)";
    card.style.border = "1px solid rgba(255,255,255,0.06)";
    card.style.borderRadius = "12px";
    card.style.padding = "16px";
    card.style.display = "flex";
    card.style.flexDirection = "column";
    card.style.gap = "8px";

    const rateColor = s.attention_rate >= 85 ? "var(--accent)" : s.attention_rate >= 75 ? "#F5B83D" : "#FF5C5C";

    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:700; font-size:14px; color:#FFF;">${code}</span>
        <span style="font-weight:700; font-size:15px; color:${rateColor}; font-family:var(--font-heading);">${s.attention_rate}% Focused</span>
      </div>
      <div style="font-size:12px; color:var(--text-muted);">${s.subject} &bull; ${s.faculty}</div>
      <div style="background:rgba(255,255,255,0.08); height:6px; border-radius:3px; overflow:hidden; margin-top:4px;">
        <div style="background:${rateColor}; width:${s.attention_rate}%; height:100%;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); margin-top:4px;">
        <span>📱 ${s.distracted_phone || 0} Phones</span>
        <span>😴 ${s.distracted_sleep || 0} Sleeping</span>
        <span>👀 ${s.distracted_away || 0} Away</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderSubjectCatalog(data) {
  const container = document.getElementById("tt-faculty-directory");
  if (!container || !data || !data.subjects_catalog) return;

  container.innerHTML = "";
  Object.values(data.subjects_catalog).forEach(sub => {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.justifyContent = "space-between";
    row.style.alignItems = "center";
    row.style.padding = "12px 16px";
    row.style.background = "var(--bg-section)";
    row.style.border = "var(--border-subtle)";
    row.style.borderRadius = "8px";

    const subjectTaught = sub.subject_taught || sub.title || sub.code;
    row.innerHTML = `
      <div>
        <div style="font-weight:700; color:#FFF; font-size:13px;">${sub.code} - ${sub.title}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">${sub.faculty} &bull; ${sub.lab_room}</div>
      </div>
      <div style="text-align:right;">
        <span style="color:var(--accent); font-size:12px; font-weight:600; background:rgba(197, 248, 42, 0.1); border:1px solid rgba(197, 248, 42, 0.25); padding:4px 10px; border-radius:6px; display:inline-block;">📚 ${subjectTaught}</span>
      </div>
    `;
    container.appendChild(row);
  });
}
