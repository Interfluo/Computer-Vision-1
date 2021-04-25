# Method 2 - Drawing bounding boxes and finding center from frame to frame
import cv2


def bounding_boxes(filename, n_boxes):
    cap = cv2.VideoCapture(filename)
    n = 0
    while cap.isOpened():
        ret, image = cap.read()
        if ret is True:
            print("frame:", n)
            n += 1
            image = cv2.resize(image, (3840 // 4, 2160 // 4))
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = cv2.blur(img, (5, 5))
            for i in range(n_boxes):
                contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    center = (int(x + 0.5 * w), int(y + 0.5 * h))
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 1)
                    cv2.circle(image, center, 3, (0, 255, 255), -1)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                    print("center for galaxy", i, ":", center)
            cv2.imshow("image", image)  # cv2.imshow("b and w", img)
            print()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


filename = "video/test_trim.mp4"
n = 50
bounding_boxes(filename, n)
