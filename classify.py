#!/usr/bin/env python
# coding: utf-8

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import signal
from imutils import paths

def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)

args = {'model': 'fashion.model', 'labelbin': 'mlb.pickle'}
testImagePaths = sorted(list(paths.list_images('examples')))

for testImagePath in testImagePaths:
    # load the image
    image = cv2.imread(testImagePath)
    output = imutils.resize(image, width=400)
 
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # load the trained convolutional neural network and the multi-label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(args["model"])
    mlb = pickle.loads(open(args["labelbin"], "rb").read())
    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    # Draw black background rectangle
    x, y, w, h = 0, 0, 165, 48
    cv2.rectangle(output, (x, x), (x + w, y + h), (0,0,0), -1)
    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 20) + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))
    # show the output image
    cv2.imshow(testImagePath, output)

cv2.waitKey(1)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

while True:
    pass