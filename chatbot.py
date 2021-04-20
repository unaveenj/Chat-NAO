#Import all required libraries
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
import os
from pandas.io.json import json_normalize
from streamlit import caching
import pandas as pd
from datetime import datetime
import sys
import pyautogui

#Disable GPU or CUDA
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu

#API Keys
API_KEY = "Enter API here"
URL = "Enter URL"
authenticator = IAMAuthenticator(API_KEY)
ta = ToneAnalyzerV3(version='2020-10-24', authenticator=authenticator)
ta.set_service_url(URL)

#Helper Functions :
#1. Function for greeting
def wishMe():
    hour=datetime.datetime.now().hour
    if hour>=0 and hour<12:
        append_file("Hello,Good Morning. Key in your input so that I can help you?")
    elif hour>=12 and hour<18:
        append_file("Hello,Good Afternoon. Key in your input so that I can help you?")
    else:
        append_file("Hello,Good Evening. Key in your input so that I can help you?")

#2. Function to extract wikipedia summary
def get_wiki_paragraph(query: str) -> str:
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0])
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0])
    return summary

#3.Speech to text function
def takecomand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        st.write("answer please....")
        audio=r.listen(source)
        try:
            text=r.recognize_google(audio)
            st.write("You  said :",text)
            r.adjust_for_ambient_noise(source,duration=1)
            # res = ta.tone(text)
            # st.json(res.result)

        except:
            st.write("Please say again ..")
    return text

#4. Face mask detector function
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

#Side bar chooser
st.sidebar.header("Chatbot service")
st.sidebar.text("Choose what you want from the chatbot")
categories = st.sidebar.radio('Case Categories: ',('Tone analyser','Chat NAO!','Wiki NAO!','Mask NAO!'))
nao_IP = st.sidebar.text_input('Enter NAO IP Address here:','')
my_bar = st.sidebar.progress(0)
if st.sidebar.button('Connect'):
    append_file('tsop')
    os.system(f'start cmd.exe /k "cd D:\\NUS Onedrive\OneDrive - National University of Singapore\FYP-Clean Version\Final NaoChatBOTv4 & conda activate python27 & python nao_connector.py {nao_IP}"')
    with st.spinner('Connection to NAO ....'):
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
    st.sidebar.success("NAO will speak on successfull connection.")

if st.sidebar.button('Stop'):
    st.sidebar.error("NAO connection terminated!")
    append_file('tsop')

if st.sidebar.button("Restart App"):
    screenWidth, screenHeight = pyautogui.size()
    pyautogui.moveTo(660, 1060)
    time.sleep(2)
    pyautogui.moveTo(660, 1000)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'c')
    pyautogui.write('streamlit run chatbot.py', interval=0.25)
    pyautogui.press('enter')
if st.sidebar.button("Quit App"):
    screenWidth, screenHeight = pyautogui.size()
    pyautogui.moveTo(660, 1060)
    time.sleep(2)
    pyautogui.moveTo(660, 1000)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'c')

#Main webpage
# if categories=='Connect NAO':
#     img = Image.open("nao_bg.png")
#     st.image(img, use_column_width=True)
#     nao_IP = st.text_input('Enter NAO IP Address here:','')
#     my_bar = st.progress(0)
#     if st.button('Connect'):
#         os.system(f'start cmd.exe /k "cd D:\\NUS Onedrive\OneDrive - National University of Singapore\FYP-Clean Version\Final NaoChatBOTv4 & conda activate python27 & python nao_connector.py {nao_IP}"')
#         with st.spinner('Transferring Data to NAO ....'):
#             for percent_complete in range(100):
#                 time.sleep(0.01)
#                 my_bar.progress(percent_complete + 1)
#         st.success("NAO will speak on successfull connection.")
#     if st.button('Stop'):
#         st.error("NAO connection terminated!")
#         append_file('tsop')


if categories=='Wiki NAO!':
    # caching.clear_cache()
    @st.cache
    def welcome_to_wikipedia():
        append_file(f"Welcome to Wikipedia with NAO. Search for your topics and ask your questions. I will answer them to my best ability!")
        # This function will only be run the first time it's called

    @st.cache(allow_output_mutation=True)
    def get_qa_pipeline() -> Pipeline:
        qa = pipeline("question-answering")
        return qa

    try:
        img = Image.open("wiki_nao.png")
        st.image(img, use_column_width=True)
        # welcome_to_wikipedia()


        st.title(" Wikipedia FAQ")

        qa = pipeline("question-answering")
        input = st.text_input("Search for your wikipedia topic: ", "")
        if input == "":
            append_file("Search for your wikipedia topic ")
            st.warning("Empty field")
        else:
            st.success('Wikipidea article extracted. Ask your questions below !')
            # st.warning(wikipedia.summary(input,setences =2))
            wiki_para = get_wiki_paragraph(input)
            st.write(wiki_para)
            question = st.text_input("Ask your questions here :", "")
            if question == "":
                st.warning("Empty question")
                append_file(f"Ask a question based on the {input} wikipedia topic")
            else:
                qa = get_qa_pipeline()
                with st.spinner('Generating Answer...'):
                    result2 = qa(question=question, context=wiki_para)
                append_file(f"You asked me {question}. The answer is ^wait(animations/Stand/Gestures/Hey_1) {result2['answer']}")
                # time.sleep(4)
                # append_file(result2['answer'])
                # time.sleep(3)
                st.write(result2['answer'])
    except:
        st.warning("Try another question please !")
        append_file(" Try asking me a question")

elif categories == 'Tone analyser':

    ignored_emotions = ["Confident","Tentative","Analytical"]
    img = Image.open("nao_bg.png")
    st.image(img, use_column_width=True)
    st.write("""
    ## Welcome to tone analyzer. 
    ### Tell me how was your day? 
    """)
    # placeholder = st.empty()
    input = st.text_input("Enter your text here", "")
    # click_clear = st.button('Click here to Clear!', key=1)
    # if click_clear:
    #     input = st.text_input('text', value='', key=1)
    if input =="":
        st.warning("Please type something!")

        # st.success("Success! Nao will respond shortly:)")
    if input !="":
        with st.spinner('Loading Model into Memory...'):
            emotions = ta.tone(input).get_result()
            time.sleep(2)

        # st.write(emotions["Sad"])
        if len(input)<8:
            st.warning("Please type more")
        else:
            try:
                st.success("NAO will respond shortly!")
                # st.success("NAO will respond shortly!")
                tones_data = json_normalize(data=emotions['sentences_tone'], record_path='tones')
                tones_data = tones_data.sort_values(by=['score'], ascending=False)
                st.dataframe(tones_data)
                fig = go.Figure(go.Bar(
                    x=tones_data['score'],
                    y=tones_data['tone_name'],
                    orientation='h'))
                st.plotly_chart(fig)
                for emotion in tones_data['tone_name'].tolist():
                    if emotion in ignored_emotions:
                        continue
                    else:
                        append_file(f'I have sensed  {emotion} in you!')
                        time.sleep(2)
                        if emotion == 'Anger' or emotion == 'Fear':
                            joke = pyjokes.get_joke(language='en', category='all')
                            joke = joke + "    huh huh! huh huh!"
                            # append_file(f"Let me tell you a joke to calm you down !  {joke} . Hope you liked the joke !")
                            append_file(f"Let me tell you a joke to calm you down !  ^wait(animations/Stand/Gestures/Hey_1) {joke} . Hope you liked the joke !")

                            time.sleep(2)
                            st.write(joke)
                            # time.sleep(4)
                            # append_file(joke)
                            # time.sleep(6)



                    # append_file(' and \n')
            except KeyError:
                # st.success("NAO will respond shortly!")
                st.warning("You can input more sentences for me to analyse!")
                tones_data = json_normalize(data=emotions['document_tone'], record_path='tones')
                tones_data = tones_data.sort_values(by=['score'], ascending=False)
                st.dataframe(tones_data)
                fig = go.Figure(go.Bar(
                    x=tones_data['score'],
                    y=tones_data['tone_name'],
                    orientation='h'))
                st.plotly_chart(fig)
                for emotion in tones_data['tone_name'].tolist():
                    if emotion in ignored_emotions:
                        continue
                    else:
                        append_file(f'I have sensed  {emotion} in you!')
                        time.sleep(2)
                        if emotion == 'Anger' or emotion == 'Fear':
                            # time.sleep(2)
                            # append_file("Let me tell you a joke to calm you down !")
                            # time.sleep(2)
                            # joke = pyjokes.get_joke(language='en', category='all')
                            # joke = joke + "    huh huh! huh huh!"

                            # time.sleep(4)
                            joke = pyjokes.get_joke(language='en', category='all')
                            joke = joke + "    huh huh!"
                            # append_file(f"Let me tell you a joke to calm you down !  {joke} . Hope you liked the joke !")
                            append_file(f"Let me tell you a joke to calm you down !  ^wait(animations/Stand/Gestures/Hey_1) {joke} . Hope you liked the joke !")
                            # st.write(joke)
                            # append_file("ha ha ha ha")

                    # append_file(' and \n')

elif categories== 'Chat NAO!':

    try:
        img = Image.open("chat_nao.png")
        st.image(img, use_column_width=True)
        choice = st.text_input("What do you want to ask NAO?","")
        choice = choice.lower()
        if choice =="" :
            # welcome_to_chatbot()
            st.warning("Type your question")
        elif 'news' in choice:
            append_file("Let me give you the top 5 breaking news for today!")
            url = ('https://newsapi.org/v2/top-headlines?'
                   'sources=bbc-news&'
                   'apiKey=Enter NEWS API key')
            response = requests.get(url)
            response = response.json()
            counter = 5
            for i in range(5):
                st.write(f"### Headlines {i+1}")
                time.sleep(3)
                st.write("Title:"+response["articles"][i]['title'])
                st.write("Description:" + response["articles"][i]['description'])
                append_file(f"Headlines {i+1}")
                time.sleep(3)
                append_file(response["articles"][i]['title'])
                time.sleep(3)
        elif 'joke' in choice or 'jokes' in choice:
            joke = pyjokes.get_joke(language='en',category='all')
            st.write(joke)
            append_file("Hey let me tell you a joke!")
            append_file(joke)

        # elif 'song' in choice or 'music' in choice:
        #     pass
        elif 'Type here' in choice:
            st.write()
        elif 'hi' in choice:
            st.write("Hello !")
            wishMe()

        else:
            app_id = "RKVG28-X4QQHAQ2LK"
            client = wolframalpha.Client("RKVG28-X4QQHAQ2LK")
            res = client.query(choice)
            answer = next(res.results).text
            append_file(f"You asked me {choice} ")
            append_file("The answer for that is ")
            append_file(answer)
            st.text(answer)
    except:
        st.warning("Try another question please !")
        append_file(" Try asking me a question")



elif categories == "Mask NAO!":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disable gpu
    #Cache
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

    append_file("Welcome to Mask Now ! To prevent the spread of Covid 19 , please wear a mask")
    # Web Image
    img = Image.open("mask_nao.png")
    st.image(img, use_column_width=True)


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
    DIR = 'defaulters'
    img_counter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    while run:
        counter = 0

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        img = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=400)
        # frame = imutils.resize(frame, width=600, height=800)
        
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
                if timer%40==0:
                    append_file("Thanks for wearing mask!")
                    timer = 0
                #time.sleep(2)
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

        if counter >= 1 and timer%100==0:
            now = datetime.now().time()
            append_file("Image has been captured of mask defaulters. They will be reported to the authorities. Please wear the mask")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'defaulters/image{img_counter}.jpg', frame)
            time_list.append(now.strftime("%H:%M:%S"))#Append time
            time_list.append(f"defaulters/image{img_counter}.jpg") #Append file name
            # st.error(f"More than {counter} people detected without mask! Please wear a mask! ")
            img_counter+=1
            timer = 0

        # show the output frame
        # st.error(f"More than {counter} people detected without mask! Please wear a mask! ")
        FRAME_WINDOW.image(frame)
        cv2.imshow("Mask detector",frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        # if st.button("Stop"):
        #     break
        # if st.button("Stop"):
        #     st.write(f'### {counter} people were detected without mask')
        #     break
        if key == ord("q"):
            break

    st.dataframe(time_df(time_list))
    # if counter >= 1:
    append_file("Perpetrator will be punished!")
    st.warning("Perpetrator will be punished!")
    for i in range(1, len(time_list), 2):
        st.image(time_list[i])
        counter+=1
    # st.write(time_list)
    append_file(f'{counter} instance where people were detected without mask')
    st.write(f'### {counter} instance where people were detected without mask')
    counter = 0

    cv2.destroyAllWindows()
    vs.stop()