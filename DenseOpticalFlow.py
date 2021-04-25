# Method 3 - Dense Optical Flow
import cv2
import numpy as np


def dense_optical_flow(filename):
    cap = cv2.VideoCapture(filename)

    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (3840 // 4, 2160 // 4))  # resize image 1
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while 1:
        ret, frame2 = cap.read()
        frame2 = cv2.resize(frame2, (3840 // 4, 2160 // 4))  # resize image 1
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
    return 0


filename = "video/test_trim.mp4"
dense_optical_flow(filename)
