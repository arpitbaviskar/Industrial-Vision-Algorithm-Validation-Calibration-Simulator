import cv2
from vision.camera_model import K, D

def undistort(img):
    h,w = img.shape[:2]
    new_K , roi = cv2.getOptimalNewCameraMatrix(
        K,D,(w,h),alpha=1
    )

    undistorted = cv2.undistort(
        img,K,D,None,new_K
    )
    return undistorted 