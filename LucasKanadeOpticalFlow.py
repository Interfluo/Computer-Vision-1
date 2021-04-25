# Method 4 - Lucas-Kanade Optical Flow
import numpy as np
import cv2


def lucas_kanade_optical_flow(filename):
    sf = 3

    cap = cv2.VideoCapture(filename)

    feature_params = dict(maxCorners=1000, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (1000, 3))

    ret, old_frame = cap.read()
    old_frame = cv2.resize(old_frame, (3840 // sf, 2160 // sf))  # resize image 1
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)
    n = 0
    while(1):# cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (3840 // sf, 2160 // sf))  # resize image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        n += 1
        print(n)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()
    cap.release()
    return 0


filename = "video/test_trim.mp4"
lucas_kanade_optical_flow(filename)
