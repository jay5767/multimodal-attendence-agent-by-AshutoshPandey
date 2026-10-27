import time
import os
import sys
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

artifact_dir = r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b"

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # 1. Live Session View
        page.goto("http://localhost:5000/")
        time.sleep(2)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_live_dashboard.png"), full_page=False)
        print("[OK] 1. Live Dashboard screenshot saved.")

        # 2. About View
        page.evaluate("switchView('about')")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_about.png"), full_page=False)
        print("[OK] 2. About View (Developer Photo & Credentials) screenshot saved.")

        # 3. Enrollment View
        page.evaluate("switchView('enrollment')")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_enrollment.png"), full_page=False)
        print("[OK] 3. Enrollment View screenshot saved.")

        # 4. Attendance View
        page.evaluate("switchView('attendance')")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_attendance.png"), full_page=False)
        print("[OK] 4. Attendance View screenshot saved.")

        # 5. Analytics View
        page.evaluate("switchView('analytics')")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_analytics.png"), full_page=False)
        print("[OK] 5. Analytics View screenshot saved.")

        # 6. Insights View
        page.evaluate("switchView('insights')")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_insights.png"), full_page=False)
        print("[OK] 6. AI Insights View screenshot saved.")

        # 7. History View
        page.evaluate("switchView('history')")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_history.png"), full_page=False)
        print("[OK] 7. Session History View screenshot saved.")

        # 8. Printable HTML Report
        page.goto("http://localhost:5000/reports/session/1/html")
        time.sleep(1.5)
        page.screenshot(path=os.path.join(artifact_dir, "screenshot_session_report.png"), full_page=False)
        print("[OK] 8. Printable HTML Report screenshot saved.")

        browser.close()
        print("[OK] All 8 SPA screenshots captured successfully with zero page reloads!")

if __name__ == "__main__":
    run()
