import cv2
import numpy as np

def apply_perspective(img,tilt_x=0.03, tilt_y=0.02):
    h, w = img.shape[:2]

    src_pts = np.float32([
        [0, 0],
        [w , 0],
        [w , h ],
        [0, h ]
    ])

    dst_pts = np.float32([
        [w * tilt_x, h * tilt_y],
        [w * (1 - tilt_x), h * tilt_y],
        [w * (1 - tilt_x), h * (1 - tilt_y)],
        [w * tilt_x, h * (1 - tilt_y)]
    ])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, H, (w, h))

    return warped