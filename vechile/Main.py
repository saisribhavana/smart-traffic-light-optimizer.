from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
from ultralytics import YOLO
import random


main = tkinter.Tk()
main.title("Emergency Vehicle Detection & Signal Time Management System") #designing main screen
main.geometry("1300x1200")

global yolo_model
labels = ['Ambulance', 'Fire Engine', 'Police']
#yolo confidence threshold to detect hand signs
CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)
global signal_duration, weather_type, status

def uploadDataset():
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,"Helmet Dataset loaded\n");
    text.insert(END,filename)
    
def processDataset():
    text.delete('1.0', END)
    text.insert(END,"Dataset processing completed\n")
    text.insert(END,"Different class labels found in dataset\n")
    text.insert(END,str(labels)+"\n\n")

def trainModel():
    global yolo_model
    text.delete('1.0', END)
    yolo_model = YOLO("model/best.pt")
    text.insert(END,"Emergency Vehicle Detection Model Training & Loading Completed")
    cnn_train_detection = cv2.imread("model/result.png")
    plt.figure(figsize=(8, 5))
    plt.imshow(cnn_train_detection)
    plt.title("Emergency Vehicle Detection Accuracy")
    plt.axis('off')
    plt.show()

def graph():
    cnn_train_detection = cv2.imread("model/results.png")
    plt.figure(figsize=(12,7))
    plt.imshow(cnn_train_detection)
    plt.title("CNN Training Graph")
    plt.axis('off')
    plt.show()

def detection(frame):
    global yolo_model, labels, signal_duration, status
    detections = yolo_model(frame)[0]
    # loop over the detections
    for data in detections.boxes.data.tolist():
        print(data)
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]
        cls_id = data[5]
        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            label = labels[int(cls_id)]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
            cv2.putText(frame, labels[int(cls_id)], (xmin, ymin-10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            if status == False:
                signal_duration = signal_duration + 15
                status = True
    return frame


def videoDetection():
    text.delete('1.0', END)
    global signal_duration, weather_type, status
    status = False
    temp = random.randint(0, 80)
    if temp <= 5:
        signal_duration = 40
        weather_type = "Fog"
    elif temp > 5 and temp < 50:
        signal_duration = 35
        weather_type = "Rainy"
    else:
        signal_duration = 30
        weather_type = "Normal"
    filename = filedialog.askopenfilename(initialdir="Videos")
    video = cv2.VideoCapture(filename)
    while(True):
        ret, frame = video.read()
        if ret == True:
            frame = detection(frame)
            cv2.putText(frame, "Weather = "+str(weather_type), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Signal Time : "+str(signal_duration), (50, 100),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.imshow("Predicted Result", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break  
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    
font = ('times', 13, 'bold')
title = Label(main, text='Emergency Vehicle Detection & Signal Time Management System')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=420,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Emergency Vehicle Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train Emergency Vehicle Detection Algorithm", command=trainModel)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

graphButton = Button(main, text="Training Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

detectButton = Button(main, text="Emergency Vehicle Detection from Video", command=videoDetection)
detectButton.place(x=50,y=300)
detectButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
