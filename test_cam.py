import cv2
import time

print("Testing camera 0")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open")
else:
    for i in range(10):
        ret, frame = cap.read()
        print(f"Frame {i}: {ret}, shape: {frame.shape if ret else None}")
        time.sleep(0.1)
cap.release()
