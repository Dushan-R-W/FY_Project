import os
import numpy as np 
from PIL import Image
import cv2
import pickle
import time
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

cascade_dir = os.path.join(base_dir, "cascades\data\haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_dir)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
labelsAndIds = {} # {'dushan':0, 'person's name':1, etc...}
labelsForRecog = [] 
imageArray = []  #store pixel values 


for dirpath, dirnames, filenames in os.walk(image_dir): #walking through every file in this folder
    for file in filenames:  #file is the files found inside each folder(pictures)
        if file.endswith("png") or file.endswith("jpg"): #looking for png & jpg
            path = os.path.join(dirpath, file) #dirpath is folder paths #file is file name
            #print(path)
            #rename the each persons folder to not have spaces and hold name temporarily in string label 
            label = os.path.basename(dirpath).replace(" ", "-").lower() 

            if not label in labelsAndIds: #doing this only once per folder. If label is not inside label_ids
                #the syntax is: mydict[key] = "value"
                labelsAndIds[label] = current_id  #so, labelsAndIds[dushan] = 0 ** labelsAndIds[ranmalee] = 1 ** and so on...
                current_id += 1

            temp_current_id = labelsAndIds[label] #temporary current_id because it changes in above line
           
            #The current version supports all possible conversions between “L”, “RGB” and “CMYK.” The matrix argument only supports “L” and “RGB”.
            grayScaleImage = Image.open(path).convert("L") #convert current image to grayscale

            height = 220
            width = int((height / grayScaleImage.height) * grayScaleImage.width)
            image_size = (width, height)
            final_image = grayScaleImage.resize(image_size, Image.ANTIALIAS) #Image resizing filters ANTIALIAS(high-quality filter)
                        
            #NumPy supports a much greater variety of numerical types than Python does. uint8 - Unsigned integer (0 to 255)
            NumpyArray_image = np.array(final_image, "uint8") #high-performance multidimensional array object
            #print(Numpy_image)

            #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
            faces = face_cascade.detectMultiScale(NumpyArray_image, 1.2, 5) #We use v2.CascadeClassifier.detectMultiScale() to find faces or eyes

            for (x,y,w,h) in faces: 
                cordinatesOfImage = NumpyArray_image[y:y+h, x:x+w] # getting square coordinates - (y coordinate start, y coordinate end) (x coordinate start, x coordinate end)
                imageArray.append(cordinatesOfImage)
                labelsForRecog.append(temp_current_id)



labelsAndIds_inverted = {v:k for k,v in labelsAndIds.items()}#inverting labels {"person_name":1} to {1:"person_name"} because in the below steps we seach face name by id
json_string = json.dumps(labelsAndIds_inverted)
savePath = os.path.join(base_dir, "Trained_Data", "referenceDictionary.json")
file1 = open(savePath, "w")
file1.write(json_string)
file1.close()


recognizer.train(imageArray, np.array(labelsForRecog))
recognizer.save(os.path.join(base_dir, "Trained_Data", "trainner.yml"))