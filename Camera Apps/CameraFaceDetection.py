import cv2
import sys
from os.path import dirname, join

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

# Load in a pretained deep learning framework to perform inference
protoPath = join(dirname(__file__), "deploy.prototxt")
modelPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000_fp16.caffemodel")
network = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
confidence_threshold = 0.7

# Continuously stream video from the camera to the output window
while cv2.waitKey(1) != 27: #27 is for Escape key
    has_frame, frame = source.read() #read() returns a boolean and a single frame from the video stream
    if not has_frame:
        break
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)

    # Run a model
    network.setInput(blob)
    detections = network.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]), 
                            (x_left_bottom + label_size[0], y_left_bottom + base_line),
                            (255, 255, 255), cv2.FILLED)

            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    elapsed, _ = network.getPerfProfile()
    label = "Inference time: %.2f ms" % (elapsed * 1000 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

    cv2.imshow(window_name, frame)


source.release()
cv2.destroyWindow(window_name)
