""" include basic functions:
    - resizing the video
    - cropping
    - dividing by size and time
    - trimming
    - saving
    - converting to greyscale
    - blurring
    - edge detection
    - counting frames"""
import cv2
import numpy as np


def frame_count(filename):
    """
    This function takes a the filename of a video and returns an integer containing the total frame count of the video
    *much faster than the count_frames function
    :param filename: string containing the path and filename of video
    :return: total_frames: integer value of how many total frames are in the video
    """
    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The video has a total of:", total_frames, "frames")
    return total_frames


def resize_frame(image, scaling_factor):
    """
    resizes an image by a specified scaling factor (new image will be 1/scaling_factor the size of original)
    :param image: original image to be manipulated
    :param scaling_factor: integer
    :return: rescaled image
    """
    size = (image.shape[1] // scaling_factor, image.shape[0] // scaling_factor)
    image = cv2.resize(image, size)  # resize image 1
    return image


def BGR2Grey(image):
    """
    converts BGR (blue, green, red) to grey scale (black and white)
    :param image: original image (BGR)
    :return: new image (grey scale)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def save_frame(result, image):
    """
    useful for saving a video frame by frame
    :param result: VideoCapture object
    :param image: frame to be saved
    :return: nothing, the video will be saved to a file
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # to save the video it must first be converted to BGR if it not
    result.write(image)
    return 0


filename = "video/test_trim.mp4"  # file path for desired video#
tf = frame_count(filename)
cap = cv2.VideoCapture(filename)

# create video capture object to save video to
ret, image = cap.read()  # get a frame for purposes of properly sizing the video writer object
sf = 4  # scaling factor
outfile = "video/result.avi"
fps = 30.0  # frames per second to save the video at
result = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), fps, (image.shape[1] // sf, image.shape[0] // sf))

while cap.isOpened():
    ret, image = cap.read()
    if ret is True:
        image = resize_frame(image, sf)
        image = cv2.GaussianBlur(image, (7, 7), 0)
        image = BGR2Grey(image)

        # Canny edge detection with automatic threshold calculation
        sigma = 0.33
        v = np.mean(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        image = cv2.Canny(image, 50, 200)

        cv2.imshow("image", image)
        save_frame(result, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # close video when it ends or if user presses the "q" key
            break
    else:
        break
cap.release()
result.release()
cv2.destroyAllWindows()

