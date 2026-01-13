import cv2
import glob

img = cv2.imread("E:\Industrial Vision Algorithm Validation & Calibration Simulator\caliberation\cb1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for w in range(6, 13):
    for h in range(5, 10):
        ret, _ = cv2.findChessboardCornersSB(gray, (w, h))
        if ret:
            print("FOUND checkerboard size:", (w, h))
