import asyncio
import os
import base64
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        # Launch Chromium with fake media stream pointing to user's photo or synthetic video
        browser = await p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--disable-web-security"
            ]
        )
        context = await browser.new_context(
            permissions=["camera", "microphone"],
            viewport={"width": 1400, "height": 900}
        )
        page = await context.new_page()

        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

        frame_requests = []
        def handle_request(req):
            if "/api/frame/process" in req.url:
                frame_requests.append(req.url)
        page.on("request", handle_request)

        print("[Verify] Navigating to http://localhost:5000/...")
        await page.goto("http://localhost:5000/", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        # Wait for camera stream and several frame processing cycles
        print("[Verify] Waiting 5 seconds for live frame loop and chart updates...")
        await asyncio.sleep(5)

        # Evaluate window charts and telemetry
        chart_status = await page.evaluate("""() => {
            const dashChart = window.dashboardChart;
            const engChart = window.engagementChart;
            const audChart = window.audioChart;
            return {
                hasDashChart: !!dashChart,
                dashLabelsCount: dashChart ? dashChart.data.labels.length : 0,
                dashLastScore: dashChart && dashChart.data.datasets[0].data.length > 0 ? dashChart.data.datasets[0].data[dashChart.data.datasets[0].data.length - 1] : null,
                hasEngChart: !!engChart,
                engLabelsCount: engChart ? engChart.data.labels.length : 0,
                hasAudChart: !!audChart,
                fpsText: document.getElementById('fps-val') ? document.getElementById('fps-val').innerText : '',
                gaugeScore: document.getElementById('gauge-score') ? document.getElementById('gauge-score').innerText : '',
                attMetric: document.getElementById('metric-attention') ? document.getElementById('metric-attention').innerText : '',
                movMetric: document.getElementById('metric-movement') ? document.getElementById('metric-movement').innerText : '',
                speechRatio: document.getElementById('metric-audio') ? document.getElementById('metric-audio').innerText : ''
            };
        }""")

        print("[Verify] Chart & UI status:", chart_status)
        print(f"[Verify] /api/frame/process calls captured: {len(frame_requests)}")

        # Screenshot Live Dashboard
        out_live = r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b\screenshot_live_fixed.png"
        await page.screenshot(path=out_live, full_page=True)
        print(f"[Verify] Live Dashboard screenshot saved to {out_live}")

        # Switch to Analytics tab
        await page.click("#nav-item-analytics")
        await asyncio.sleep(1.5)

        out_analytics = r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b\screenshot_analytics_fixed.png"
        await page.screenshot(path=out_analytics, full_page=True)
        print(f"[Verify] Analytics view screenshot saved to {out_analytics}")

        await browser.close()
        print("[Verify] Done!")

if __name__ == "__main__":
    asyncio.run(run())
