
import numpy as np
import cv2
from tkinter import *
import pickle
import os
import time
import string

base_dir = os.path.dirname(os.path.abspath(__file__))
cascade_dir = os.path.join(base_dir, "cascades\data\haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_dir)

filename_increment = 0
cap = cv2.VideoCapture(0)

participant_name = input("Enter a name")
print("Working.....")

working_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(working_dir, "images")
path = os.path.join(image_dir, participant_name)
os.mkdir(path)
img_name = participant_name + str(filename_increment) + ".png"

while (filename_increment < 50):
    ret, frame = cap.read() # Capture frame-by-frame
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray frame to use in the code
    faces = face_cascade.detectMultiScale(grayFrame, 1.16, 5)

    #putting text on the screen to count number of pics taken
    cv2.putText(frame, str(filename_increment), (30,30), cv2.QT_FONT_NORMAL, 1, (130,0, 140), 2, )

    for (x, y, w, h) in faces:

        #print(x, y, w, h)
        # Region of Interest (ROI) allows to operate a rectangular subset of the image, roi_gray is a local variable
        # getting square coordinates - (y coordinate start, y coordinate end) (x coordinate start, x coordinate end)
        roi_gray = grayFrame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        color = (255, 0, 0)  # BGR all blue color
        stroke = 2  # thickness
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke) #cv2.rectangle(image, start_point, end_point, color, thickness)

        img_name = participant_name + str(filename_increment) + ".png"
        cv2.imwrite(os.path.join(path, img_name), roi_color) #cv2.imwrite(img_item, roi_gray) #method used to save images to any storage device(filename, image)
        filename_increment += 1

        time.sleep(0.2)

    newFrame = cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()