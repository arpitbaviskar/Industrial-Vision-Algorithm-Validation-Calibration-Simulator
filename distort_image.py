import cv2
import numpy as np
from camera_model import K, D

def apply_distortion(img):
    h, w = img.shape[:2]

    new_K = K.copy()

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=D,
        R=None,
        newCameraMatrix=new_K,
        size=(w, h),
        m1type=cv2.CV_32FC1
    )

    distorted = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )

    return distorted
