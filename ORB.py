# Method 1 - using ORB
import cv2


def orb_method(filename, n_kp):
    cap1 = cv2.VideoCapture(filename)  # create a VideoCapture object and read the input file
    cap2 = cv2.VideoCapture(filename)
    for j in range(1):  # move the second video ahead by 2 frames
        ret2, frame2 = cap2.read()
    n = 0  # initialize frame count at zero
    sf = 4  # scale factor [ie. frame size = (height/sf, width/sf)]
    while cap2.isOpened():
        ret1, frame1 = cap1.read()  # capture frame by frame
        ret2, frame2 = cap2.read()
        if ret2 is True:
            frame1 = cv2.resize(frame1, (3840 // sf, 2160 // sf))  # resize image 1
            frame2 = cv2.resize(frame2, (3840 // sf, 2160 // sf))  # resize image 2
            # ------------------------------------------------------------------------------------------------ #
            # ORB (oriented fast - rotated brief) algorithm: detects key-points and computes image descriptors #
            # the general idea is to match objects from one frame to the next, and then extract depth          #
            # information. With this  it should be possible to create a depth map (at least a simple one).     #
            orb = cv2.ORB_create(nfeatures=n_kp)  # create orb object for keypoint detection
            kp1, des1 = orb.detectAndCompute(frame1, None)  # compute key-points and descriptors
            kp2, des2 = orb.detectAndCompute(frame2, None)
            kp_img1 = cv2.drawKeypoints(frame1, kp1, None, color=(200, 80, 0), flags=0)  # draw key-points
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # create bf (brute-force) matcher object
            matches = bf.match(des1, des2)  # match key-points between frames
            matches = sorted(matches, key=lambda x: x.distance)  # sort the matches
            list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]  # get matched key-point coordinate data
            list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
            for i in range(len(list_kp2)):  # plot
                x1 = list_kp1[i][0]
                y1 = list_kp1[i][1]
                x2 = list_kp2[i][0]
                y2 = list_kp2[i][1]
                dx = 5 * (x2 - x1)
                dy = 5 * (y2 - y1)
                l_thick = int(((dx ** 2 + dy ** 2) ** 0.5) / 10) + 1
                if l_thick < 10 / sf:  # filters out matches based on a distance criteria
                    cv2.arrowedLine(kp_img1, (int(x1), int(y1)), (int(x1 + dx), int(y1 + dy)),
                                    (255, 200, 120), thickness=l_thick)  # plot kept matches
            # ------------------------------------------------------------------------------------------------ #
            cv2.imshow("frame", kp_img1)
            if n % 10 == 0:
                print("frame:", n)  # print current frame
            n += 1  # update frame count
            if cv2.waitKey(1) & 0xFF == ord('q'):  # close the video when user presses "q"
                break
        else:
            break
    print("done")
    cap1.release()  # release the VideoCapture object
    cap2.release()  # release the VideoCapture object
    cv2.destroyAllWindows()  # closes all the frames
    return 0


filename = "video/test_trim.mp4"  # file path for desired video
n_kp = 500  # number of key points to detect in each frame
orb_method(filename, n_kp)
