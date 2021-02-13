# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:57:55 2020
@author: Shane
"""
import cv2
import json
import numpy as np
import pandas as pd
import math

## This takes the excel file of orientation and converts it to a pandas dataframe with multi-layer indexing in the same format/order as our json file (first ndex is frame number, second is sperm number)
orientation = pd.read_excel(r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\orientations_0.xlsx", header = None)
orientation = orientation.set_index([0,1])
orientation = orientation[2]
