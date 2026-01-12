import cv2
import numpy as np
import glob


print("OpenCV version:", cv2.__version__)


# ---------------- CONFIG ----------------
CHECKERBOARD = (6, 8)  # inner corners
MIN_VALID_IMAGES = 5
# ----------------------------------------



objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("calibration/*.jpg")

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

    print(f"{fname} → {'FOUND' if ret else 'NOT FOUND'}")

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# ----------- SAFETY CHECK ----------------
if len(objpoints) < MIN_VALID_IMAGES:
    raise RuntimeError(
        f"Not enough valid checkerboard detections: {len(objpoints)}"
    )

# ----------- CALIBRATION -----------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_shape,
    None,
    None
)

print("\nCamera Matrix (K):\n", K)
print("\nDistortion Coefficients:\n", dist)
print("\nRe-projection Error:\n", ret)