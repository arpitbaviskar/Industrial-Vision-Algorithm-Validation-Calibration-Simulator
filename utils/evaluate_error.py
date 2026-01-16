import cv2
import numpy as np
from vision.camera_model import K, D

def reprojection_error(objpoints, imgpoints, rvecs, tvecs):
    total_error = 0
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, D
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error

    return total_error / len(objpoints)