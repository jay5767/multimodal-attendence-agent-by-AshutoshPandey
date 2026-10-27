import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from modules.vision.face_engine import FaceEngine
from modules.attendance.manager import AttendanceManager
from modules.enrollment.manager import EnrollmentManager

def run_benchmarks():
    print("=" * 70)
    print("  MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT - BENCHMARK SUITE")
    print("=" * 70)

    # 1. Initialize Face Engine
    t0 = time.time()
    face_engine = FaceEngine()
    hw = face_engine.get_acceleration_info()
    init_time = time.time() - t0
    print(f"[*] Engine Initialized in {init_time:.2f}s | Provider: {hw['provider']} | Model: {hw['model']}")

    # 2. Benchmark Detection Rate & Processing FPS
    num_test_frames = 40
    print(f"\n[*] Running FPS & Detection Rate Benchmark ({num_test_frames} frames)...")

    # Generate synthetic camera frames containing face features
    test_frames = []
    for i in range(num_test_frames):
        canvas = np.full((480, 640, 3), 25, dtype=np.uint8)
        # Draw realistic face ellipse
        cx = 320 + int(np.sin(i * 0.2) * 60)
        cy = 240 + int(np.cos(i * 0.2) * 30)
        cv2.ellipse(canvas, (cx, cy), (65, 85), 0, 0, 360, (180, 210, 235), -1)
        # Eyes
        cv2.circle(canvas, (cx - 24, cy - 15), 7, (40, 40, 40), -1)
        cv2.circle(canvas, (cx + 24, cy - 15), 7, (40, 40, 40), -1)
        # Mouth
        cv2.ellipse(canvas, (cx, cy + 28), (20, 8), 0, 0, 180, (50, 50, 150), -1)
        test_frames.append(canvas)

    detected_count = 0
    start_bench = time.perf_counter()

    for frame in test_frames:
        dets = face_engine.detect_faces(frame)
        if len(dets) > 0:
            detected_count += 1

    total_bench_time = time.perf_counter() - start_bench
    avg_fps = round(num_test_frames / total_bench_time, 1)
    detection_rate = round((detected_count / num_test_frames) * 100.0, 1)

    print(f"    -> Average Inference FPS: {avg_fps} FPS")
    print(f"    -> Face Detection Rate:   {detection_rate}%")

    # 3. Benchmark Recognition Accuracy & Cosine Separation
    print("\n[*] Running Cosine Similarity & Recognition Accuracy Benchmark...")
    # Generate 4 distinct enrolled identities
    np.random.seed(42)
    enrolled_dict = {}
    for sid in range(1, 5):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        enrolled_dict[sid] = vec

    correct_matches = 0
    total_evals = 100

    for _ in range(total_evals):
        target_sid = np.random.randint(1, 5)
        clean_emb = enrolled_dict[target_sid]
        # Realistic face variation (small perturbation in 512-d latent space)
        noise = np.random.randn(512).astype(np.float32) * 0.025
        noisy_emb = clean_emb + noise
        noisy_emb /= np.linalg.norm(noisy_emb)

        matched_id, sim = face_engine.match_embedding(noisy_emb, enrolled_dict, threshold=0.45)
        if matched_id == target_sid:
            correct_matches += 1

    accuracy = round((correct_matches / total_evals) * 100.0, 1)
    print(f"    -> Enrolled Sample Recognition Accuracy: {accuracy}% (evaluated over {total_evals} noisy embeddings)")

    # 4. Impostor Rejection Rate (Unknown faces)
    unregistered_evals = 50
    rejections = 0
    for _ in range(unregistered_evals):
        unknown_vec = np.random.randn(512).astype(np.float32)
        unknown_vec /= np.linalg.norm(unknown_vec)
        matched_id, sim = face_engine.match_embedding(unknown_vec, enrolled_dict, threshold=0.45)
        if matched_id is None:
            rejections += 1

    rejection_rate = round((rejections / unregistered_evals) * 100.0, 1)
    print(f"    -> Impostor / Unknown Rejection Rate:    {rejection_rate}%")

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"  Provider:             {hw['provider']}")
    print(f"  Model:                {hw['model']}")
    print(f"  Inference Speed:      {avg_fps} FPS (Target: 20+ FPS with process skip)")
    print(f"  Face Detection Rate:  {detection_rate}%")
    print(f"  Recognition Accuracy: {accuracy}%")
    print(f"  Impostor Rejection:   {rejection_rate}%")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
