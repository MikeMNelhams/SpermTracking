## USE FILE FOR TRUTH IN FOLDER FORMAT IMPORTANT

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:43:11 2021

@author: Shane
"""
import pandas as pd
import numpy as np
import math
import motmetrics as mm


##Convert truth table to pandas
truth_file = r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_4_YOLO_NO_TRACKING_output\ground_truth.xlsx"
truth_df = pd.read_excel(truth_file,header = None,index_col=0)


hyp_file = r"C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\MDM3_B\0_4\MHT.xlsx"
hyp_df = pd.read_excel(hyp_file,header = None)

nb_tracks = 33

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

for frame in range(0,truth_df.shape[0]):
    truth_IDs = []
    hyp_IDs = []

    pos_truth = []
    pos_hyp = []
    
    for i in range(1,nb_tracks*2,2):
        if truth_df[i][frame+1] == 0:
            continue
        else:
            trackID = round((i-1)/2)
            truth_IDs.append(trackID)
            pos_truth.append([truth_df[i][frame+1],truth_df[i+1][frame+1]] )#x and y values
            
    keys = list(hyp_df.loc[hyp_df[2] == frame][0].keys()) #list of rows for that frame
    for key in keys:
        hyp_IDs.append( hyp_df[3][key])
        pos_hyp.append([hyp_df[0][key],hyp_df[1][key]])

    dists = np.sqrt(mm.distances.norm2squared_matrix(pos_truth,pos_hyp))
    frame_id = acc.update(truth_IDs,hyp_IDs,dists)
    
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
print(summary)
