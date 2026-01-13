import cv2
import numpy as np
from distort_image import apply_distortion
from perspective_warp import apply_perspective
from undistort_image import undistort

img = cv2.imread(r"E:\Industrial Vision Algorithm Validation & Calibration Simulator\caliberation\cb1.jpg")
assert img is not None, "Image not found"

distorted = apply_distortion(img)
warped = apply_perspective(distorted)
corrected = undistort(warped)

comparison = np.hstack([
    cv2.resize(img, (400,300)),
    cv2.resize(warped, (400,300)),
    cv2.resize(corrected, (400,300))
])

cv2.imshow("Original | Distorted+Warped | Corrected", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
