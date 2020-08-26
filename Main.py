from tkinter import *
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))

#*****Functions******#
def FacesOption():
    print("Running... please wait")
    exec(open(os.path.join(path, "FaceDetection&Recognition.py"), encoding="utf8").read())
    print("Face Recognition finished")

def AddOption():
    print("Running... please wait")
    exec(open(os.path.join(path, "save_50_faces.py"), encoding="utf8").read())
    print("Adding face finished")

def TrainOption():
    print("Running... please wait")
    exec(open(os.path.join(path, "DataTraining.py"), encoding="utf8").read())
    print("Training finished")

#*****Options******#
while True:
    print("")
    print("*******--Menu--*********")
    print("'1' - Face Recognition")
    print("'2' - Add new face")
    print("'3' - Run training")
    print("'4' - Exit")
    print("************************")
    print("")

    try:
        UserInput = int(input("Enter option:"))
    except:
        print("Invalid input")

    if(UserInput == 1):
        FacesOption()

    if(UserInput == 2):
        AddOption()

    if(UserInput == 3):
        TrainOption()

    if(UserInput == 4):
        sys.exit()



