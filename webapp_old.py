from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
import streamlit as st
import speech_recognition as sr
from PIL import Image
import text2emotion as te
import matplotlib.pyplot as plt
import plotly.express as px
import time
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import quote_generator
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
# #Authentication
# API_KEY = "GZT7tn7BpHaDVYvDqj1dLX5i4RhHgTVYRfgQ3D_Xfvrv"
# URL = "https://api.au-syd.tone-analyzer.watson.cloud.ibm.com/instances/b404c5c1-fbcd-409c-97c5-326612bcbdab"
# authenticator = IAMAuthenticator(API_KEY)
# ta = ToneAnalyzerV3(version='2020-10-24', authenticator=authenticator)
# ta.set_service_url(URL)

# @st.cache
# def welcome_to_tone_analyser():
#     # This function will only be run the first time it's called
#     append_file("Welcome to NAO tone analyzer ! Type into the field and I will analyse your emotion.")
#
# @st.cache
# def welcome_to_chatbot():
#     # This function will only be run the first time it's called
#     append_file("Welcome to ASK Now! Ask me anything by typing in the input field !")
# @st.cache
# def welcome_to_wikipedia():
#     append_file(f"Wikiepedia summary on the topic has been extracted and seen on display!")
#     # This function will only be run the first time it's called


def wishMe():
    hour=datetime.datetime.now().hour
    if hour>=0 and hour<12:
        append_file("Hello,Good Morning. Key in your input so that I can help you?")
    elif hour>=12 and hour<18:
        append_file("Hello,Good Afternoon. Key in your input so that I can help you?")
    else:
        append_file("Hello,Good Evening. Key in your input so that I can help you?")

def get_wiki_paragraph(query: str) -> str:
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0])
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0])
    return summary


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




max_width = 1500
padding_top=10
padding_right=10
padding_left=10
padding_bottom=10
#css
# st.markdown(
#         f"""
# <style>
#     .reportview-container .main .block-container{{
#         max-width: {max_width}px;
#         padding-top: {padding_top}rem;
#         padding-right: {padding_right}rem;
#         padding-left: {padding_left}rem;
#         padding-bottom: {padding_bottom}rem;
#     }}
#     .reportview-container .main {{
#     }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )

#Title of web app
# st.write("""
# # NAO ChatBot Webapp
# """)

# append_file("Hi welcome to NAO Chatbot! There are different services in this webapp. Feel free to use them!")

#side bar

st.sidebar.header("Chatbot service")
st.sidebar.text("Choose what you want from the chatbot")
categories = st.sidebar.radio('Case Categories: ',('Tone analyser','NAO Ask me!','Mask NAO!'))
quotes = ['albert_einstein','motivational','mahatma_gandhi','bill_gates','steve_jobs']
quote_func = [quote_generator.albert_einstein_quotes,quote_generator.motivational_quotes,quote_generator.mahatma_gandhi_quotes,quote_generator.bill_gates_quotes,quote_generator.steve_jobs_quotes]
#main app

if categories == 'Tone analyser':

    img = Image.open("nao_bg.png")
    st.image(img, use_column_width=True)
    st.write("""
    ## Welcome to tone analyzer. 
    ### Tell me how was your day? 
    """)
    # welcome_to_tone_analyser()
    # append_file("Welcome to NAO tone analyzer ! Type into the field and I will analyse your emotion.")
    # append_file("Welcome to NAO tone analyzer ! Type into the field and I will analyse your emotion.")
    input = st.text_input("Enter your text here", "")
    if input =="":
        st.warning("Please type something!")
        # append_file("Welcome to NAO tone analyzer ! Type into the field and I will analyse your emotion.")

    # Web Image
    #     append_file("You can speak to me also by pressing the speaking button !")

    # input="I am sad"
    # if st.button("Submit"):
    # result = input.title()
    my_bar = st.progress(0)
        # st.success("Success! Nao will respond shortly:)")
    if input !="":
        with st.spinner('Loading Model into Memory...'):
            emotions = te.get_emotion(input)
            time.sleep(2)
        with st.spinner('Transferring Data to NAO ....'):
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
        # st.write(emotions["Sad"])
        nao_emotions = sorted(emotions, key=emotions.get, reverse=True)[:2]
        st.write(nao_emotions[0])
        append_file(f"Hey I see that you are feeling {nao_emotions[0]}........  ")

        if nao_emotions[0]=="Sad":
            append_file("Let me play a    song for you!")
        elif nao_emotions[0]=="Happy":
            append_file("Let me play a song for you!")
        elif nao_emotions[0]=="Angry":
            append_file("Let me tell a joke......... to cheer you up.......... ")
            joke = pyjokes.get_joke(language='en',category='all')
            append_file(joke)
            st.write(joke)
        elif nao_emotions[0]=='Fear':
            i = random.randint(0,4)
            quote = quote_func[i]()
            append_file(f'Hey, dont be worries. {quote}')

        if emotions[nao_emotions[1]]>0.3:
            if nao_emotions[0] == "Sad":
                append_file("Let me play a    song for you!")
            elif nao_emotions[0] == "Happy":
                append_file("Let me play a song for you!")
            elif nao_emotions[0] == "Angry":
                append_file("Let me tell a joke......... to cheer you up.......... ")
                joke = pyjokes.get_joke(language='en', category='all')
                append_file(joke)
                st.write(joke)


        st.success("Success! Nao will respond shortly:)")
        fig = px.pie(values=list(emotions.values()), names=list(emotions.keys()), title="Emotion Analysis")
        if sum(list(emotions.values())) is not 0:
            st.plotly_chart(fig)


    st.write("### Press Speak to speak instead")
    try:
        if st.button("Speak"):
            voice_input = takecomand()
            with st.spinner('Loading Model into Memory...'):
                emotions = te.get_emotion(voice_input)
                time.sleep(2)
            with st.spinner('Transferring Data to NAO ....'):
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)

            nao_emotions = sorted(emotions, key=emotions.get, reverse=True)[:2]
            if (emotions[nao_emotions[1]]>0.3):
                append_file(f"Hey I see that you are feeling {nao_emotions[0]} and {nao_emotions[1]} ")
                if "Sad" in nao_emotions or "Happy" in nao_emotions:
                    append_file("Let me play a song for you!")
            else:
                append_file(f"Hey I see that you are feeling {nao_emotions[0]} ")
                if nao_emotions[0]=="Sad" or nao_emotions[0]=="Happy":
                    append_file("Let me play a song for you!")
                elif nao_emotions[0]=="Angry":
                    append_file("Let me tell a joke to calm down")
                    append_file(pyjokes.get_joke(language='en',category='all'))

            st.success("Success! Nao will respond shortly:)")
            fig2 = px.pie(values=list(emotions.values()), names=list(emotions.keys()), title="Emotion Analysis")
            if sum(list(emotions.values())) is not 0:
                st.plotly_chart(fig2)
    except:
        append_file("Try again !")

elif categories == "Mask NAO!":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disable gpu

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
    run =st.checkbox("Click here for Real-time detection!")
    FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(0)
    vs = VideoStream(src=0).start()
    counter = 0
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
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            if mask>withoutMask:
                label ="Mask"
                append_file("Thanks for wearing mask!")
                time.sleep(2)
            else:
                label = "No Mask"
                counter = counter+1
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

        if counter >=2 :
            append_file("More than two people not wearing mask")
            cv2.imwrite('image.jpg', frame) #Capture the culprits
            # st.error(f"More than {counter} people detected without mask! Please wear a mask! ")
            break

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
    if counter>=2:
        append_file("Perpetrator will be punished!")
        st.warning("Perpetrator will be punished!")
        st.image('image.jpg')
    append_file(f'{counter} people were detected without mask')
    st.write(f'### {counter} people were detected without mask')
    counter = 0

    cv2.destroyAllWindows()
    vs.stop()

else:
    try:
        img = Image.open("ask_nao.png")
        st.image(img, use_column_width=True)
        choice = st.text_input("What do you want NAO to do ?","")
        if choice =="" :
            # welcome_to_chatbot()
            st.warning("Type your question")
        elif "wikipedia" in choice or "wiki" in choice:
            # welcome_to_wikipedia()
            @st.cache(allow_output_mutation=True)
            def get_qa_pipeline() -> Pipeline:
                qa = pipeline("question-answering")
                return qa
            st.title(" Wikipedia FAQ")
            with st.spinner('Loading Model into Memory...'):
                qa = pipeline("question-answering")
            input = st.text_input("Search for your wikipedia topic: ","" )
            if input=="":
                append_file("Search for your wikipedia topic ")
                st.warning("Empty field")
            else:

                # st.warning(wikipedia.summary(input,setences =2))
                wiki_para = get_wiki_paragraph(input)
                st.write(wiki_para)
                question = st.text_input("Ask your questions here :","")
                if question =="":
                    st.warning("Empty question")
                    append_file(f"Ask a question based on the {input} wikipedia topic")
                else:
                    qa= get_qa_pipeline()
                    result2 = qa(question=question, context=wiki_para)
                    append_file(result2['answer'])
                    st.write(result2['answer'])
        elif 'news' in choice:
            append_file("Let me give you the top 3 breaking news for today!")
            url = ('https://newsapi.org/v2/top-headlines?'
                   'sources=bbc-news&'
                   'apiKey=03dff64123dc4a45b1f1d81758079a60')
            response = requests.get(url)
            response = response.json()
            counter = 5
            for i in range(3):
                st.write(f"### Headlines {i+1}")
                st.write("Title:"+response["articles"][i]['title'])
                st.write("Description:" + response["articles"][i]['description'])
                append_file(f"Headlines {i+1}")
                append_file(response["articles"][i]['title'])
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
            append_file(wishMe())
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
        append_file("Sorry I do not understand! Please try another question")









