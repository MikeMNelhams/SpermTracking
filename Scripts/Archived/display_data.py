# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:57:55 2020

@author: Shane
"""
import cv2
import json
import numpy as np


#Improvements : making sperm head centers appear gradually
#Following code allows the information on bounding boxes, sperm head centers and sperm id to be displayed on video 

#Load data (change path to json files)
with open(r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp57\cover1_1_YOLO_NO_TRACKING_output\centroids_with_meta.json", "r") as read_file:
    data = json.load(read_file)
    
#Load video (change path to video)    
cap = cv2.VideoCapture(r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp57\cover1_1.avi")


success,img = cap.read()

##Formatting and scaling successive sperm head positions to display
def drawCenters(img,i):
    for i in range(0,len(data["centroids"])): # number of frames
        for j in range(0,len(data["centroids"][i])): #number of sperms in frame
            center = tuple(x*data["extra_information"]["resize_factor"] for x in data["centroids"][i][j]["center"])
            cv2.circle(img,(np.float32(center[0]),np.float32(center[1])),1,(255,0,0),1)


#Formatting and scaling bounding boxes to display (with ID)
def drawBox(img,i):
    for j in range(0,len(data["centroids"][i])):
        bbox = tuple([x*data["extra_information"]["resize_factor"] for x in data["centroids"][i][j]["bbox"]])
        cv2.rectangle(img,(np.float32(bbox[0]),np.float32(bbox[1])),((np.float32(bbox[0]+bbox[2])),np.float32(bbox[1]+bbox[3])),(255,0,255),3,1)
        cv2.putText(img ,str(data["centroids"][i][j]["id"]), (np.float32(bbox[0]-10),np.float32(bbox[1]-10)) , cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)

i = 0 
#Displaying bounding box according to video frame       
while True:
    
    success,img = cap.read()
    drawCenters(img,i)
    drawBox(img,i)
    i +=1
    
    cv2.imshow("Tracking",img)
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
cv2.destroyAllWindows()
