import requests

print("[*] Downloading socket.io.min.js...")
sio = requests.get("https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js", timeout=15).text
with open(r"C:\Users\Ashutoshpandey\.gemini\antigravity\scratch\multimodal-attendance-agent\static\js\socket.io.min.js", "w", encoding="utf-8") as f:
    f.write(sio)

print("[*] Downloading chart.min.js...")
cjs = requests.get("https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js", timeout=15).text
with open(r"C:\Users\Ashutoshpandey\.gemini\antigravity\scratch\multimodal-attendance-agent\static\js\chart.min.js", "w", encoding="utf-8") as f:
    f.write(cjs)

print("[✓] Both libraries downloaded locally into static/js/")
