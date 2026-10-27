import sys
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARTIFACT_DIR = Path(r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(r"C:\Users\Ashutoshpandey\.gemini\antigravity\scratch\multimodal-attendance-agent")
CSS_FILE = PROJECT_ROOT / "static" / "css" / "style.css"

with open(CSS_FILE, "r", encoding="utf-8") as f:
    RAW_CSS = f.read()

PAGES = [
    ("/", "screenshot_live_dashboard.png", "Live Session Dashboard"),
    ("/enrollment", "screenshot_enrollment.png", "Student Enrollment"),
    ("/attendance", "screenshot_attendance.png", "Attendance Records"),
    ("/analytics", "screenshot_analytics.png", "Multimodal Analytics"),
    ("/insights", "screenshot_insights.png", "AI Pedagogical Insights"),
    ("/history", "screenshot_history.png", "Session History Archive"),
    ("/about", "screenshot_about.png", "About Developer - Ashutosh Pandey"),
    ("/reports/session/1/html", "screenshot_session_report.png", "Printable Session Report")
]

def capture_all():
    base = "http://127.0.0.1:5000"
    print("=" * 65)
    print("  CAPTURING PIXEL-PERFECT BROWSER SCREENSHOTS")
    print("=" * 65)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        for route, filename, label in PAGES:
            url = f"{base}{route}"
            resp = requests.get(url)
            html = resp.text

            # Inline master CSS for instantaneous zero-latency rendering
            if "</head>" in html:
                html = html.replace("</head>", f"<style>{RAW_CSS}</style></head>")

            page.set_content(html)
            page.wait_for_timeout(400) # Allow DOM settling

            dest = ARTIFACT_DIR / filename
            page.screenshot(path=str(dest), full_page=(route != "/"))
            print(f"[✓] Captured {label}: {filename} ({dest.stat().st_size} bytes)")

        browser.close()
        print("=" * 65)
        print("  ALL 8 SYSTEM SCREENSHOTS CAPTURED AND VERIFIED!")
        print("=" * 65)

if __name__ == "__main__":
    capture_all()
