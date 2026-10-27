/**
 * Chart.js Telemetry Initializer for Multimodal Analytics & Live Dashboard
 * Dark theme with Electric Lime line, soft gradient fill, red drop markers, and lime peak markers.
 */

window.dashboardChart = null;
window.engagementChart = null;
window.audioChart = null;

let recentScoresHistory = [75, 76, 78, 80];

document.addEventListener("DOMContentLoaded", () => {
  initAllClassroomCharts();
});

function initAllClassroomCharts() {
  initDashboardChart();
  initAnalyticsCharts();
}

function initDashboardChart() {
  const dashCanvas = document.getElementById("dashboard-engagement-chart");
  if (!dashCanvas) return;

  const ctx = dashCanvas.getContext("2d");
  const limeGradient = ctx.createLinearGradient(0, 0, 0, 180);
  limeGradient.addColorStop(0, "rgba(197, 248, 42, 0.40)");
  limeGradient.addColorStop(1, "rgba(197, 248, 42, 0.0)");

  const initialLabels = ["00:05", "00:10", "00:15", "00:20", "00:25", "00:30"];
  const initialData = [74, 76, 79, 81, 78, 80];
  const initialBase = [74, 75, 76, 77, 77, 78];

  window.dashboardChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: initialLabels,
      datasets: [
        {
          label: "Live Engagement (%)",
          data: initialData,
          borderColor: "#C5F82A",
          backgroundColor: limeGradient,
          borderWidth: 2.5,
          fill: true,
          tension: 0.35,
          pointBackgroundColor: "#C5F82A",
          pointBorderColor: "#0A0A0A",
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 7
        },
        {
          label: "Rolling Baseline (%)",
          data: initialBase,
          borderColor: "rgba(255, 255, 255, 0.25)",
          borderWidth: 1.5,
          borderDash: [4, 4],
          fill: false,
          tension: 0.2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      plugins: {
        legend: {
          display: true,
          labels: { color: "#9CA3A0", font: { family: "Inter", size: 11 }, boxWidth: 12 }
        },
        tooltip: {
          backgroundColor: "#1A1D1A",
          titleColor: "#FFFFFF",
          bodyColor: "#C5F82A",
          borderColor: "rgba(197, 248, 42, 0.3)",
          borderWidth: 1,
          padding: 8,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#9CA3A0", font: { family: "Inter", size: 10 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#9CA3A0", font: { family: "Inter", size: 10 } }
        }
      }
    }
  });
}

function initAnalyticsCharts() {
  const engCanvas = document.getElementById("engagement-chart");
  const audCanvas = document.getElementById("audio-chart");
  if (!engCanvas || !audCanvas) return;

  const engCtx = engCanvas.getContext("2d");
  const audCtx = audCanvas.getContext("2d");

  const limeGradient = engCtx.createLinearGradient(0, 0, 0, 320);
  limeGradient.addColorStop(0, "rgba(197, 248, 42, 0.35)");
  limeGradient.addColorStop(1, "rgba(197, 248, 42, 0.0)");

  const initialLabels = ["00:30", "01:00", "01:30", "02:00", "02:30", "03:00", "03:30", "04:00"];
  const initialEngagement = [76, 78, 80, 82, 61, 58, 79, 84];
  const initialBaseline = [76, 77, 78, 79, 75, 71, 73, 76];
  
  const pointColors = initialEngagement.map((val, idx) => {
    if (idx === 4 || idx === 5) return "#FF5C5C"; // Drop
    if (idx === 7) return "#C5F82A"; // Peak
    return "#FFFFFF";
  });

  const pointRadii = initialEngagement.map((val, idx) => {
    return (idx === 4 || idx === 7) ? 7 : 4;
  });

  window.engagementChart = new Chart(engCtx, {
    type: "line",
    data: {
      labels: initialLabels,
      datasets: [
        {
          label: "Class Engagement (%)",
          data: initialEngagement,
          borderColor: "#C5F82A",
          backgroundColor: limeGradient,
          borderWidth: 3,
          fill: true,
          tension: 0.35,
          pointBackgroundColor: pointColors,
          pointBorderColor: "#0A0A0A",
          pointBorderWidth: 2,
          pointRadius: pointRadii,
          pointHoverRadius: 8
        },
        {
          label: "Rolling Baseline (%)",
          data: initialBaseline,
          borderColor: "rgba(255, 255, 255, 0.25)",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      plugins: {
        legend: {
          labels: {
            color: "#9CA3A0",
            font: { family: "Inter", size: 12 }
          }
        },
        tooltip: {
          backgroundColor: "#1A1D1A",
          titleColor: "#FFFFFF",
          bodyColor: "#C5F82A",
          borderColor: "rgba(197, 248, 42, 0.3)",
          borderWidth: 1,
          padding: 12,
          cornerRadius: 10,
          callbacks: {
            label: function(context) {
              const val = context.parsed.y;
              if (context.dataIndex === 4) return `Engagement: ${val}% (⚠️ Attention Drop -23%)`;
              if (context.dataIndex === 7) return `Engagement: ${val}% (⚡ Interaction Peak)`;
              return `Engagement: ${val}%`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#9CA3A0", font: { family: "Inter" } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#9CA3A0", font: { family: "Inter" } }
        }
      }
    }
  });

  // Audio Speech Ratio Bar Chart
  const audioSpeechRatios = [0.18, 0.22, 0.28, 0.35, 0.04, 0.06, 0.42, 0.72];
  const barColors = audioSpeechRatios.map(r => r > 0.65 ? "#C5F82A" : (r < 0.10 ? "#FF5C5C" : "rgba(255, 255, 255, 0.45)"));

  window.audioChart = new Chart(audCtx, {
    type: "bar",
    data: {
      labels: initialLabels,
      datasets: [
        {
          label: "Speech Activity Ratio",
          data: audioSpeechRatios,
          backgroundColor: barColors,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      plugins: {
        legend: {
          labels: { color: "#9CA3A0", font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          backgroundColor: "#1A1D1A",
          titleColor: "#FFFFFF",
          bodyColor: "#C5F82A",
          borderColor: "rgba(197, 248, 42, 0.3)",
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#9CA3A0", font: { family: "Inter" } }
        },
        y: {
          min: 0,
          max: 1.0,
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#9CA3A0", font: { family: "Inter" } }
        }
      }
    }
  });
}

/**
 * Live Telemetry Recorder: Appends dynamic observations in real time
 */
window.recordLiveTelemetryPoint = function(score, speechRatio = 0.20, isDrop = false, isPeak = false, customLabel = null) {
  const now = new Date();
  const timeLabel = customLabel || `${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
  const roundedScore = Math.round(Number(score) || 75);

  recentScoresHistory.push(roundedScore);
  if (recentScoresHistory.length > 15) recentScoresHistory.shift();

  // Calculate rolling baseline
  const baseline = Math.round(recentScoresHistory.reduce((a, b) => a + b, 0) / recentScoresHistory.length);

  // Dynamic point color
  let ptColor = "#FFFFFF";
  let ptRadius = 4;
  if (isDrop || (recentScoresHistory.length >= 3 && roundedScore < baseline - 15)) {
    ptColor = "#FF5C5C";
    ptRadius = 7;
    isDrop = true;
  } else if (isPeak || roundedScore > 85) {
    ptColor = "#C5F82A";
    ptRadius = 6;
  }

  // Update Dashboard Chart (Live view)
  if (window.dashboardChart && window.dashboardChart.data) {
    const d = window.dashboardChart.data;
    d.labels.push(timeLabel);
    d.datasets[0].data.push(roundedScore);
    d.datasets[1].data.push(baseline);

    if (!Array.isArray(d.datasets[0].pointBackgroundColor)) {
      d.datasets[0].pointBackgroundColor = d.datasets[0].data.map(() => "#C5F82A");
      d.datasets[0].pointRadius = d.datasets[0].data.map(() => 4);
    }
    d.datasets[0].pointBackgroundColor.push(ptColor);
    d.datasets[0].pointRadius.push(ptRadius);

    if (d.labels.length > 20) {
      d.labels.shift();
      d.datasets[0].data.shift();
      d.datasets[1].data.shift();
      d.datasets[0].pointBackgroundColor.shift();
      d.datasets[0].pointRadius.shift();
    }
    window.dashboardChart.update("none");
  }

  // Update Analytics Chart (Analytics view)
  if (window.engagementChart && window.engagementChart.data) {
    const e = window.engagementChart.data;
    e.labels.push(timeLabel);
    e.datasets[0].data.push(roundedScore);
    e.datasets[1].data.push(baseline);

    if (!Array.isArray(e.datasets[0].pointBackgroundColor)) {
      e.datasets[0].pointBackgroundColor = e.datasets[0].data.map(() => "#FFFFFF");
      e.datasets[0].pointRadius = e.datasets[0].data.map(() => 4);
    }
    e.datasets[0].pointBackgroundColor.push(ptColor);
    e.datasets[0].pointRadius.push(ptRadius);

    if (e.labels.length > 25) {
      e.labels.shift();
      e.datasets[0].data.shift();
      e.datasets[1].data.shift();
      e.datasets[0].pointBackgroundColor.shift();
      e.datasets[0].pointRadius.shift();
    }
    window.engagementChart.update("none");
  }

  // Update Audio Bar Chart
  if (window.audioChart && window.audioChart.data) {
    const a = window.audioChart.data;
    const ratioVal = Math.min(1.0, Math.max(0.0, Number(speechRatio) || 0.0));
    a.labels.push(timeLabel);
    a.datasets[0].data.push(ratioVal);

    const bColor = ratioVal > 0.65 ? "#C5F82A" : (ratioVal < 0.10 ? "#FF5C5C" : "rgba(255,255,255,0.45)");
    if (!Array.isArray(a.datasets[0].backgroundColor)) {
      a.datasets[0].backgroundColor = a.datasets[0].data.map(() => "rgba(255,255,255,0.45)");
    }
    a.datasets[0].backgroundColor.push(bColor);

    if (a.labels.length > 25) {
      a.labels.shift();
      a.datasets[0].data.shift();
      a.datasets[0].backgroundColor.shift();
    }
    window.audioChart.update("none");
  }

  // Update Trend Badge on Dashboard
  const trendBadge = document.getElementById("timeline-trend-badge");
  if (trendBadge) {
    if (isDrop) {
      trendBadge.innerHTML = `<span style="color: #FF5C5C;">⚠️ Attention Drop (${roundedScore}%)</span>`;
    } else if (roundedScore >= 75) {
      trendBadge.innerHTML = `<span style="color: #C5F82A;">⚡ High Engagement (${roundedScore}%)</span>`;
    } else {
      trendBadge.innerHTML = `<span style="color: #F5B83D;">● Moderate Engagement (${roundedScore}%)</span>`;
    }
  }
};

/**
 * Resets charts when a new classroom session starts
 */
window.resetChartsForNewSession = function() {
  recentScoresHistory = [];
  [window.dashboardChart, window.engagementChart, window.audioChart].forEach(chart => {
    if (chart && chart.data) {
      chart.data.labels = [];
      chart.data.datasets.forEach(ds => {
        ds.data = [];
        if (Array.isArray(ds.pointBackgroundColor)) ds.pointBackgroundColor = [];
        if (Array.isArray(ds.pointRadius)) ds.pointRadius = [];
        if (Array.isArray(ds.backgroundColor)) ds.backgroundColor = [];
      });
      chart.update();
    }
  });
};
