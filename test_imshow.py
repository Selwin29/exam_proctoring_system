import cv2
import time
import numpy as np

print("Testing imshow...")
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "TEST", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

try:
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Test", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Test", img)
    print("imshow called.")
    for _ in range(50):
        cv2.waitKey(100)
    print("done waitkey")
except Exception as e:
    print("Error:", e)

cv2.destroyAllWindows()
