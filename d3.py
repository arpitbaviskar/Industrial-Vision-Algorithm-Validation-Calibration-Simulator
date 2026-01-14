import cv2
import numpy as np
import glob
import os
from run_d2 import SQUARE_SIZE_MM
# ============================
# Load Day 1 & Day 2 outputs
camera_matrix = np.load(r"E:\Industrial Vision Algorithm Validation & Calibration Simulator\calibration\camera_matrix.npy")
dist_coeffs = np.load(r"E:\Industrial Vision Algorithm Validation & Calibration Simulator\calibration\dist_coeffs.npy")
H = np.load(r"E:\Industrial Vision Algorithm Validation & Calibration Simulator\calibration\homography.npy")

# ============================
def pixel_to_world(pt, H):
    p = np.array([pt[0], pt[1], 1.0])
    w = H @ p
    w /= w[2]
    return w[:2]

def measurement_error(measured_mm, actual_mm):
    error = abs(measured_mm - actual_mm)
    percent = (error / actual_mm) * 100
    return error, percent

def repeatability_stats(values):
    v = np.array(values)
    return {
        "mean_mm": np.mean(v),
        "std_dev_mm": np.std(v),
        "min_mm": np.min(v),
        "max_mm": np.max(v),
        "range_mm": np.ptp(v)
    }

# ============================
# Configuration
CHECKERBOARD = (8, 6)
known_length_mm = SQUARE_SIZE_MM

img_dir = r"E:\Industrial Vision Algorithm Validation & Calibration Simulator\calibration"
img_paths = sorted(glob.glob(os.path.join(img_dir, "cb*.jpg")))

if len(img_paths) == 0:
    raise RuntimeError("No images found")

measurements = []

# ============================
# MAIN LOOP (THIS IS DAY 3)
for path in img_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    # Undistort (Day 1)
    img = cv2.undistort(img, camera_matrix, dist_coeffs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
    if not ret:
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Pick two consistent real corners
    p1_pixel = corners[0][0]
    p2_pixel = corners[1][0]   # same row, known spacing

    # Pixel → world (Day 2)
    p1_world = pixel_to_world(p1_pixel, H)
    p2_world = pixel_to_world(p2_pixel, H)

    measured_mm = np.linalg.norm(p2_world - p1_world)
    measurements.append(measured_mm)

    print(f"{os.path.basename(path)} → {measured_mm:.3f} mm")

# ============================
# RESULTS
stats = repeatability_stats(measurements)
abs_err, pct_err = measurement_error(stats["mean_mm"], known_length_mm)

print("\n--- Day 3 Results ---")
print("Images used:", len(measurements))
print("Mean Measurement:", stats["mean_mm"])
print("Std Deviation:", stats["std_dev_mm"])
print("Absolute Error:", abs_err)
print("Percent Error:", pct_err)
print("Repeatability Stats:", stats)
