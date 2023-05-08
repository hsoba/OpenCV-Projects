import cv2
import sys

# Default camera device index
# 0 is the system default and on Mac points to iPhone
# 1 is the internal Mac camera
camera_device_index = 1

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

# Continuously stream video from the camera to the output window
while cv2.waitKey(1) != 27: #27 is for Escape key
    has_frame, frame = source.read() #read() returns a boolean and a single frame from the video stream
    if not has_frame:
        break
    cv2.imshow(window_name, frame)


source.release()
cv2.destroyWindow(window_name)
