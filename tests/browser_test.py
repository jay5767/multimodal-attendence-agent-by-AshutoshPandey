import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARTIFACT_DIR = Path(r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def run_browser_verification():
    log("[*] Starting Headless Browser via Playwright (channel=msedge)...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--no-sandbox"]
        )
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            permissions=["camera", "microphone"]
        )
        page = context.new_page()

        # 1. Live Session Dashboard
        log("[*] Navigating to Live Session Dashboard...")
        page.goto("http://127.0.0.1:5000/", wait_until="domcontentloaded", timeout=15000)
        time.sleep(2)
        live_shot = ARTIFACT_DIR / "screenshot_live_dashboard.png"
        page.screenshot(path=str(live_shot))
        log(f"[✓] Captured Live Session Dashboard: {live_shot.name}")

        # 2. Enrollment Page
        log("[*] Navigating to Student Enrollment Page...")
        page.goto("http://127.0.0.1:5000/enrollment", wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)
        enroll_shot = ARTIFACT_DIR / "screenshot_enrollment.png"
        page.screenshot(path=str(enroll_shot))
        log(f"[✓] Captured Enrollment Page: {enroll_shot.name}")

        # 3. Analytics Page
        log("[*] Navigating to Multimodal Analytics Page...")
        page.goto("http://127.0.0.1:5000/analytics", wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)
        analytics_shot = ARTIFACT_DIR / "screenshot_analytics.png"
        page.screenshot(path=str(analytics_shot))
        log(f"[✓] Captured Analytics Page: {analytics_shot.name}")

        # 4. About Developer Page
        log("[*] Navigating to About Developer Page...")
        page.goto("http://127.0.0.1:5000/about", wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)
        about_shot = ARTIFACT_DIR / "screenshot_about.png"
        page.screenshot(path=str(about_shot))
        log(f"[✓] Captured About Page: {about_shot.name}")

        # 5. Printable HTML Report
        log("[*] Navigating to Session Report...")
        page.goto("http://127.0.0.1:5000/reports/session/1/html", wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)
        report_shot = ARTIFACT_DIR / "screenshot_report.png"
        page.screenshot(path=str(report_shot))
        log(f"[✓] Captured Printable HTML Report: {report_shot.name}")

        browser.close()
        log("\n[✓] ALL 5 BROWSER VERIFICATION SCREENSHOTS SAVED SUCCESSFULLY!")

if __name__ == "__main__":
    run_browser_verification()
