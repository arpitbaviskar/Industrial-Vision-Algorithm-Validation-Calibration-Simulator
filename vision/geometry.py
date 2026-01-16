import cv2
import numpy as np
import glob
import os
from vision.camera_model import load_camera_model


def compute_homography(image_dir, checkerboard=(8, 6), square_size_mm=10.0):
    """
    Day 2: Compute planar homography from checkerboard to world plane

    Parameters
    ----------
    image_dir : str
        Directory containing calibration images
    checkerboard : tuple
        Inner checkerboard corners (cols, rows)
    square_size_mm : float
        Physical square size in mm

    Returns
    -------
    H : np.ndarray
        3x3 homography matrix
    """

    print("=== Day 2: Geometry Calibration (Homography) ===")

    # Load camera model (Day 1 output)
    K, dist = load_camera_model(image_dir)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "cb*.jpg")))
    if len(image_paths) == 0:
        raise RuntimeError("No checkerboard images found for homography")

    H = None
    used_image = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        # Undistort using calibration
        img = cv2.undistort(img, K, dist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, checkerboard)
        print(f"{os.path.basename(path)} → {'FOUND' if ret else 'NOT FOUND'}")

        if not ret:
            continue

        # Subpixel refinement
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Build world coordinates (mm)
        objp = np.zeros((checkerboard[0] * checkerboard[1], 2), np.float32)
        objp[:, 0] = np.tile(
            np.arange(checkerboard[0]), checkerboard[1]
        ) * square_size_mm
        objp[:, 1] = np.repeat(
            np.arange(checkerboard[1]), checkerboard[0]
        ) * square_size_mm

        # Compute homography
        H, _ = cv2.findHomography(corners.reshape(-1, 2), objp)
        used_image = path
        break

    if H is None:
        raise RuntimeError("Failed to compute homography from any image")

    np.save(os.path.join(image_dir, "homography.npy"), H)

    print("Homography computed using:", os.path.basename(used_image))
    print("Saved homography to:", image_dir)

    return H
