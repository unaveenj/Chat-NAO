import os
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
import streamlit as st
import speech_recognition as sr
from PIL import Image
import text2emotion as te
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.video import VideoStream
import datetime
from transformers import pipeline, Pipeline
import wikipedia
import requests
import pyjokes
import wolframalpha
from test import append_file
import random
import os
from pandas.io.json import json_normalize
from datetime import datetime
import pandas as pd
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu

append_file("Welcome to Mask Now ! To prevent the spread of Covid 19 , please wear a mask")
# Web Image
img = Image.open("mask_nao.png")
st.image(img, use_column_width=True)

@st.cache(allow_output_mutation=True)
def time_list_gen():
    time_list = []
    return time_list
@st.cache(allow_output_mutation=True)
def time_df(time_list):
    df = pd.DataFrame(columns=["Time_Caught","File_Name"])
    for i in range(0, len(time_list), 2):
        df = df.append({"Time_Caught": time_list[i], "File_Name": time_list[i + 1]}, ignore_index=True)
    return df

@st.cache(hash_funcs={cv2.dnn_Net: hash})
def load_face_detector_and_model():
    prototxt_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weights_path = os.path.sep.join(["face_detector",
                                     "res10_300x300_ssd_iter_140000.caffemodel"])
    cnn_net = cv2.dnn.readNet(prototxt_path, weights_path)

    return cnn_net


@st.cache(allow_output_mutation=True)
def load_cnn_model():
    cnn_model = load_model("mask_detector.model")

    return cnn_model


with st.spinner('Loading Model into Memory...'):
    net = load_face_detector_and_model()
    model = load_cnn_model()
    time.sleep(1)

st.write("# Instructions:")
st.write("1.Check the box to run real time mask detection")
st.write("2.Uncheck and check the box again to run again !")
run = st.checkbox("Click here for Real-time detection!")
FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)
vs = VideoStream(src=0).start()
counter = 0
timer = 0
#Time report of ppl not wearing mask
df = pd.DataFrame(columns=["Time_Caught","File_Name"])
time_list = time_list_gen()
img_counter = 0
now = datetime.now().time()
while run:
    counter = 0

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600, height=800)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, net, model)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        timer+=1
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        if mask > withoutMask:
            label = "Mask"
            append_file("Thanks for wearing mask!")
            time.sleep(2)
        else:
            label = "No Mask"
            counter = counter + 1
        # if counter == 1:
        #     st.error(f"More than {counter} people detected without mask! Please wear a mask! ")
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    if counter >= 1 and timer%75==0:
        append_file("More than two people not wearing mask")
        cv2.imwrite(f'image{img_counter}.jpg', frame)
        time_list.append(now.strftime("%H:%M:%S"))#Append time
        time_list.append(f"image{img_counter}.jpg") #Append file name
        # st.error(f"More than {counter} people detected without mask! Please wear a mask! ")
        img_counter+=1
        timer = 0

    # show the output frame
    # st.error(f"More than {counter} people detected without mask! Please wear a mask! ")
    FRAME_WINDOW.image(frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    # if st.button("Stop"):
    #     break
    # if st.button("Stop"):
    #     st.write(f'### {counter} people were detected without mask')
    #     break

st.dataframe(time_df(time_list))
# if counter >= 1:
append_file("Perpetrator will be punished!")
st.warning("Perpetrator will be punished!")
for i in range(1, len(time_list), 2):
    st.image(time_list[i])
    counter+=1
# st.write(time_list)
append_file(f'{counter} people were detected without mask')
st.write(f'### {counter} instance of people were detected without mask')
counter = 0

cv2.destroyAllWindows()
vs.stop()