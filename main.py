import cv2
import numpy as np
import glob
import os

print("OpenCV version:", cv2.__version__)
# ---------------- CONFIG ----------------
CHECKERBOARD = (8, 6)        # ✅ CORRECT inner corners
MIN_VALID_IMAGES = 5
# ----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "calibration")
os.makedirs(CALIB_DIR, exist_ok=True)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(CALIB_DIR, "cb*.jpg"))

img_shape = None

# ------------ DETECTION LOOP -------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(f"{os.path.basename(fname)} → {'FOUND' if ret else 'NOT FOUND'}")

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# ----------- SAFETY CHECK ----------------
if len(objpoints) < MIN_VALID_IMAGES:
    raise RuntimeError(
        f"Not enough valid checkerboard detections: {len(objpoints)}"
    )

# ----------- CALIBRATION -----------------
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

# ----------- SAVE OUTPUTS ----------------
np.save(os.path.join(CALIB_DIR, "camera_matrix.npy"), K)
np.save(os.path.join(CALIB_DIR, "dist_coeffs.npy"), dist)

print("\nSaved calibration files to:", CALIB_DIR)
