import cv2
import numpy as np
import glob
import os


def run_calibration(image_dir, checkerboard=(8, 6), min_valid_images=5):
    print("=== Day 1: Camera Calibration ===")
    print("OpenCV version:", cv2.__version__)

    calib_dir = image_dir

    # Prepare object points
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    # ✅ ONLY calibration images
    images = glob.glob(os.path.join(calib_dir, "cb*.jpg"))
    if len(images) == 0:
        raise RuntimeError("No calibration images found (cb*.jpg)")

    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCornersSB(
            gray,
            checkerboard,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        print(f"{os.path.basename(fname)} → {'FOUND' if ret else 'NOT FOUND'}")

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) < min_valid_images:
        raise RuntimeError(
            f"Not enough valid checkerboard detections: {len(objpoints)}"
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_shape,
        None,
        None
    )

    print("\nCamera Matrix (K):\n", K)
    print("\nDistortion Coefficients:\n", dist)
    print("\nRMS Reprojection Error:", rms)

    np.save(os.path.join(calib_dir, "camera_matrix.npy"), K)
    np.save(os.path.join(calib_dir, "dist_coeffs.npy"), dist)

    print("\nSaved calibration files to:", calib_dir)

    return K, dist, rms
