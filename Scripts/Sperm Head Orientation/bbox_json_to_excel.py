# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:21:05 2021

@author: Shane
"""
# Import Statements
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def export_to_excel(data_input):
    x_values, y_values, frame_values, w_values, h_values = [], [], [], [], []

    for i in range(0, len(data_input["centroids"])):  # number of frames
        for j in range(0, len(data_input["centroids"][i])):  # number of sperms in frame
            x_values.append(data_input["centroids"][i][j]["bbox"][0])
            y_values.append(data_input["centroids"][i][j]["bbox"][1])
            w_values.append(data_input["centroids"][i][j]["bbox"][2])
            h_values.append(data_input["centroids"][i][j]["bbox"][3])
            frame_values.append(i)
    dict_temp = {"frame" : frame_values, "x_min" : x_values, "y_min" : y_values, "width" : w_values, "height" : h_values}
    df = pd.DataFrame(dict_temp)
    df.to_excel(r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\vid_0.xlsx",index = False,header = False )
        
    return 0

print('Importing video Data...')
with open(r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\centroids_with_meta.json", "r") as read_file:
    data = json.load(read_file)
print('Data import completed.')


export_to_excel(data)  # Export data to a spreadsheet
