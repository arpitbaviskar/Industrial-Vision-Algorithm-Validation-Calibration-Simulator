import cv2
import numpy as np
import glob
import os

# ---------------- CONFIG ----------------
CHECKERBOARD = (8, 6)        # inner corners
SQUARE_SIZE_MM = 10.0        # change to your real square size
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "calibration")

camera_matrix = np.load(os.path.join(CALIB_DIR, "camera_matrix.npy"))
dist_coeffs = np.load(os.path.join(CALIB_DIR, "dist_coeffs.npy"))

# Load ONE reference image
img_path = os.path.join(CALIB_DIR, "cb13.jpg")
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("Reference image not found")

# Undistort using Day 1 results
img = cv2.undistort(img, camera_matrix, dist_coeffs)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect checkerboard
ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)
if not ret:
    raise RuntimeError("Checkerboard not detected")

# Subpixel refinement
corners = cv2.cornerSubPix(
    gray, corners, (11,11), (-1,-1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

# -------- World coordinates (mm) --------
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 2), np.float32)
objp[:, 0] = np.tile(np.arange(CHECKERBOARD[0]), CHECKERBOARD[1]) * SQUARE_SIZE_MM
objp[:, 1] = np.repeat(np.arange(CHECKERBOARD[1]), CHECKERBOARD[0]) * SQUARE_SIZE_MM

# -------- Homography computation --------
H, _ = cv2.findHomography(corners.reshape(-1,2), objp)

print("Homography Matrix:\n", H)

# -------- Save homography --------
np.save(os.path.join(CALIB_DIR, "homography.npy"), H)
print("Saved homography to calibration/homography.npy")

# -------- Visual sanity check --------
vis = img.copy()
cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ret)
cv2.imshow("Checkerboard Detection (Day 2)", cv2.resize(vis, (800,600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
