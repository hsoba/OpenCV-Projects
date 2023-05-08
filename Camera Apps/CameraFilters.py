import cv2
import sys
import numpy

# Preset filter values
PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

corner_params = dict(   maxCorners = 500,
                        qualityLevel = 0.2,
                        minDistance = 15,
                        blockSize = 9)

# Default camera device index
# 0 is the system default and on Mac points to iPhone
# 1 is the internal Mac camera
camera_device_index = 1

# Default camera filter
image_filter = PREVIEW
alive = True

# Check if there's a command line specification that overrides the default camera index
if len(sys.argv) > 1:
    camera_device_index = sys.argv[1]

# Create a video capture object
source = cv2.VideoCapture(camera_device_index)

# Check if video capture object is successfully instantiated
if source.isOpened() == False:
    print("Error opening video stream")
else:
    print("Video stream successful")

window_name = 'Camera Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
result = None

# Continuously stream video from the camera to the output window
while alive:
    has_frame, frame = source.read() #read() returns a boolean and a single frame from the video stream
    if not has_frame:
        break

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13,13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **corner_params)

        if corners is not None:
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 0, 255), 1)
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 145, 150)

    cv2.imshow(window_name, result)

    # Exit window
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(window_name)
