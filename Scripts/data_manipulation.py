# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:37:37 2020

@author: Shane
"""
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#I used the path for my PC i expect there is a better way to do it from github
#if anyone knows how to do that
with open(r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\centroids_with_meta.json", "r") as read_file:
    data = json.load(read_file)


#Data is in form of a dict

#First classsification is either data["centroids"] or data["extra_information"]
#The key "extra_information" only contains the resize factor
# The key "centroids" provides a list where the index is the desired frame number of the video

#For video cover_0_0 there are 301 frames 

#For each frame number i, there is a set number of sperms
# len(data["centroids"][i]) will give the number of sperms in frame i (this changes across frames)

#For each sperm detected in a given frame, there is a dictionary containing
#the following info :
#- a box containing sperm head ("bbox"), 
#- the position of the center of the sperm head ("center"),
# the class("class"),
#(which i expect is whether the detected object is a sperm or not because it looks like it is always set to 1)
#- whether it was interpolated (Set false for all) ("interpolated"),
#- A sperm ID (which for now is unique for all sperms) ("id")

#My understanding is that once we have recognised two sperm points over two frames
# to be the same sperm, the one in the latest frame will be set to have the same ID and have an interpolated value of True

#Overall, the relevant data is in form
#data["centroids"][frame_number][sperm_number]["desired_info_on_sperm"]

#Therefore : the position of the center of the first sperm's head in the first frame will be
print(data["centroids"][0][0]["center"])

x_values = []
y_values = []
frame_values=[]
#Plotting all positions of center of sperm heads for video 0
for i in range(0,len(data["centroids"])): # number of frames
    for j in range(0,len(data["centroids"][i])): #number of sperms in frame
        x_values.append(data["centroids"][i][j]["center"][0])
        y_values.append(data["centroids"][i][j]["center"][1])
        frame_values.append(i)

#2D  - Various positions of all sperm heads for all frames
#plt.scatter(x_values,y_values,s=1)

#3D - adding frame dimension (can be converted to time with FPS value)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x_values,y_values,frame_values,color='r',s=1)
ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Frame Axes')

plt.show()

