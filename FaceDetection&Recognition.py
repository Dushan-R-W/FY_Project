import numpy as np
import cv2
from tkinter import *
import pickle
import os
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
cascade_dir = os.path.join(base_dir, "cascades\data\haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_dir)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(base_dir, "Trained_Data", "trainner.yml"))

#loads the saved json file
loadPath = os.path.join(base_dir, "Trained_Data", "referenceDictionary.json")
labelsAndIds = {"1": "name"}

file1Stream = open(loadPath, "r")
jsonFile = file1Stream.read()
labelsAndIds = json.loads(jsonFile)

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read() # Capture frame-by-frame
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray frame to use in the code
    faces = face_cascade.detectMultiScale(grayFrame, 1.2, 5)

    #show quit text
    cv2.putText(frame, "Q - Exit", (30,30), cv2.QT_FONT_NORMAL, 0.6, (130,0, 140), 2, )
    cv2.rectangle(frame, (30,40), (120,40), (117,0,125), 2)

    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        # Region of Interest (ROI) allows to operate a rectangular subset of the image, roi_grayImage is a local variable
        # getting square coordinates - (y coordinate start, y coordinate end) (x coordinate start, x coordinate end)
        roi_grayImage = grayFrame[y:y + h, x:x + w]
        roi_colorImage = frame[y:y + h, x:x + w]

        #fps = cap.get(cv2.cv2.CAP_PROP_FPS)
        #print(fps)

        matchID, conf = recognizer.predict(roi_grayImage)
        matchIDString = str(matchID)

        if conf <=50:
            #print(matchID)
            #print("type:", type(labelsAndIds_inverted))
            #print(labelsAndIds_inverted[matchID])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = (labelsAndIds[matchIDString]) + " - " + str(round(conf))
            color = (213, 255, 0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke)
        

        color = (255, 0, 0)  # BGR all blue color
        stroke = 2  # thickness
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke) #cv2.rectangle(image, start_point, end_point, color, thickness)

    # display resulting frame in colour
    newFrame = cv2.imshow('frame', frame)
    
    #******exit code******#
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()