import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b"

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # 1. Capture Enrollment page showing enrolled students
        print("Navigating to enrollment...")
        page.goto("http://localhost:5000/enrollment", wait_until="networkidle")
        time.sleep(1.5)
        page.screenshot(path=f"{ARTIFACT_DIR}/screenshot_enrollment_verified.png", full_page=True)
        print("Captured screenshot_enrollment_verified.png")

        # 2. Toggle Anon Mode and capture masked view
        print("Toggling Anonymous Mode...")
        page.click("#btn-anon-toggle")
        time.sleep(1.0)
        page.screenshot(path=f"{ARTIFACT_DIR}/screenshot_anon_mode_verified.png", full_page=True)
        print("Captured screenshot_anon_mode_verified.png")

        # 3. Navigate to Live Session
        print("Navigating to Live Session...")
        page.click("#nav-item-live")
        time.sleep(1.5)
        page.screenshot(path=f"{ARTIFACT_DIR}/screenshot_live_dashboard_verified.png", full_page=True)
        print("Captured screenshot_live_dashboard_verified.png")

        browser.close()

if __name__ == "__main__":
    run()
