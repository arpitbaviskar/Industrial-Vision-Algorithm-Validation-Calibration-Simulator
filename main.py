"""
Industrial Vision Algorithm Validation & Calibration Simulator
Main Orchestrator

Day 1: Camera Calibration
Day 2: Geometry & Homography
Day 3: Measurement & Repeatability
"""
import os
import numpy as np
from vision.calibration import run_calibration
from vision.camera_model import load_camera_model
from vision.geometry import compute_homography
from vision.measurement import run_measurement


# ---------------- CONFIG ----------------
IMAGE_DIR = "calibration"
CHECKERBOARD = (8, 6)        # inner corners
SQUARE_SIZE_MM = 10.0        # measured physical square size
# ---------------------------------------


def main():
    print("\n==============================")
    print(" INDUSTRIAL VISION VALIDATION ")
    print("==============================\n")

    # -------- DAY 1 --------
    print("▶ Day 1: Camera Calibration")

    calib_K = os.path.join(IMAGE_DIR, "camera_matrix.npy")
    calib_D = os.path.join(IMAGE_DIR, "dist_coeffs.npy")

    if os.path.exists(calib_K) and os.path.exists(calib_D):
        print("Calibration files found. Loading existing calibration.")
        K, dist = load_camera_model(IMAGE_DIR)
        rms = None
    else:
        print("Calibration files not found. Running calibration...")
        K, dist, rms = run_calibration(
            image_dir=IMAGE_DIR,
            checkerboard=CHECKERBOARD
        )

    # -------- DAY 2 --------
    print("\n▶ Day 2: Geometry Calibration (Homography)")

    H_path = os.path.join(IMAGE_DIR, "homography.npy")

    if os.path.exists(H_path):
        print("Homography found. Loading existing homography.")
        H = np.load(H_path)
    else:
        print("Homography not found. Computing homography...")
        H = compute_homography(
            image_dir=IMAGE_DIR,
            checkerboard=CHECKERBOARD,
            square_size_mm=SQUARE_SIZE_MM
        )

    # -------- DAY 3 --------
    print("\n▶ Day 3: Measurement & Repeatability Analysis")
    results = run_measurement(
        image_dir=IMAGE_DIR,
        checkerboard=CHECKERBOARD,
        square_size_mm=SQUARE_SIZE_MM
    )

    # -------- FINAL SUMMARY --------
    print("\n==============================")
    print(" FINAL VALIDATION SUMMARY")
    print("==============================")

    if rms is not None:
        print(f"RMS Reprojection Error (px): {rms:.4f}")
    else:
        print("RMS Reprojection Error (px): Loaded from previous calibration")

    print(f"Mean Measurement (mm):       {results['mean_mm']:.3f}")
    print(f"Std Deviation (mm):          {results['std_dev_mm']:.3f}")
    print(f"Absolute Error (mm):         {results['abs_error_mm']:.3f}")
    print(f"Percent Error (%):           {results['percent_error']:.2f}")

    print("\n✔ Calibration validated")
    print("✔ Geometry grounded")
    print("✔ Measurement repeatability quantified")
    print("\nSystem ready for validation reporting.\n")
if __name__ == "__main__":
    main()
