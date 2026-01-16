import cv2
import numpy as np
import glob
import os
from vision.camera_model import load_camera_model


def pixel_to_world(pt, H):
    p = np.array([pt[0], pt[1], 1.0])
    w = H @ p
    w /= w[2]
    return w[:2]


def run_measurement(image_dir, checkerboard=(8, 6), square_size_mm=10.0):
    print("=== Day 3: Measurement & Repeatability ===")

    K, dist = load_camera_model(image_dir)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "cb*.jpg")))
    if len(image_paths) == 0:
        raise RuntimeError("No images found for measurement")

    # World coordinates (mm)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 2), np.float32)
    objp[:, 0] = np.tile(np.arange(checkerboard[0]), checkerboard[1]) * square_size_mm
    objp[:, 1] = np.repeat(np.arange(checkerboard[1]), checkerboard[0]) * square_size_mm

    measurements = []

    cols = checkerboard[0]
    row = 0
    idx1 = row * cols + 0
    idx2 = row * cols + 5
    known_length_mm = 5 * square_size_mm

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.undistort(img, K, dist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, checkerboard)
        if not ret:
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        if idx2 >= len(corners):
            continue

        # ✅ Per-image homography (correct placement)
        H, _ = cv2.findHomography(corners.reshape(-1, 2), objp)
        if H is None:
            continue

        p1_pixel = corners[idx1][0]
        p2_pixel = corners[idx2][0]

        p1_world = pixel_to_world(p1_pixel, H)
        p2_world = pixel_to_world(p2_pixel, H)

        measured_mm = np.linalg.norm(p2_world - p1_world)
        measurements.append(measured_mm)

        print(f"{os.path.basename(path)} → {measured_mm:.3f} mm")

    if len(measurements) == 0:
        raise RuntimeError("No valid measurements produced")

    measurements = np.array(measurements)

    mean_mm = np.mean(measurements)
    std_mm = np.std(measurements)
    abs_err = abs(mean_mm - known_length_mm)
    pct_err = (abs_err / known_length_mm) * 100

    return {
        "mean_mm": mean_mm,
        "std_dev_mm": std_mm,
        "abs_error_mm": abs_err,
        "percent_error": pct_err,
        "min_mm": np.min(measurements),
        "max_mm": np.max(measurements),
        "range_mm": np.ptp(measurements),
        "num_samples": len(measurements),
    }
