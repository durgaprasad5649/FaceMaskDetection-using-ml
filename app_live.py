# Import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
from pygame import mixer
import tensorflow as tf

# Print TensorFlow version
print(tf.__version__)

# Initialize mixer and load sound for alert
mixer.init()
sound = mixer.Sound('mixkit-security-facility-breach-alarm-994.wav')

# Load face detection model
# Load face detection model with raw string for file paths
prototxtPath = r"C:\Users\durga\Desktop\FaceMaskDetection\face_detector\deploy.prototxt"  # Path to deploy.prototxt.txt
weightsPath = r"C:\Users\durga\Desktop\FaceMaskDetection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"  # Path to Caffe model
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Define custom DepthwiseConv2D layer without the 'groups' argument
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.initializers import GlorotUniform

def custom_depthwise_conv2d(**kwargs):
    kwargs.pop('groups', None)  # Remove the invalid 'groups' argument
    return DepthwiseConv2D(**kwargs)

# Load the model with custom objects
maskNet = tf.keras.models.load_model(
    'fmd_model.h5',
    custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d, 'GlorotUniform': GlorotUniform}
)

# Define the mask_detection_prediction function
def mask_detection_prediction(frame, faceNet, maskNet):
    # Detect faces in the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    locs = []
    preds = []

    # Loop through detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(face)[0]
            locs.append((startX, startY, endX, endY))
            preds.append((mask, withoutMask))

    return (locs, preds)

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Main loop for processing video frames
while True:
    # Capture frame, resize it, and detect faces/masks
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = mask_detection_prediction(frame, faceNet, maskNet)

    # Display bounding boxes and labels for each face detected
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine label and color
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
            print("Normal")
        else:
            label = "No Mask"
            color = (0, 0, 255)
            sound.play()
            print("Alert!!!")

        # Display label with probability and draw bounding box
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break loop if 'q' key is pressed
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
