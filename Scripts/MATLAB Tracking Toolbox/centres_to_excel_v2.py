# Import Statements
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def export_to_excel(data_input, cover_number):
    x_values, y_values, frame_values = [], [], []

    for i in range(0, len(data_input["centroids"])):  # number of frames
        for j in range(0, len(data_input["centroids"][i])):  # number of sperms in frame
            x_values.append(data_input["centroids"][i][j]["center"][0])
            y_values.append(data_input["centroids"][i][j]["center"][1])
            frame_values.append(i)
    detection_nums = [x for x in range(1,len(x_values)+1)]
    dict_temp = {"detection number": detection_nums, "frame" : frame_values, "x_min" : x_values, "y_min" : y_values}
    df = pd.DataFrame(dict_temp)
    df.to_excel(r"D:\Uni work\Engineering Mathematics Work\MDM3\Mojo\mojo_sperm_tracking_data_bristol\tp49\cover{}_YOLO_NO_TRACKING_output\vid{}.xlsx".format(cover_number,cover_number),index = False,header = False )
        
    return 0

print('Importing video Data...')
cover_number = "0_6"
with open(r"D:\Uni work\Engineering Mathematics Work\MDM3\Mojo\mojo_sperm_tracking_data_bristol\tp49\cover{}_YOLO_NO_TRACKING_output\centroids_with_meta.json".format(cover_number), "r") as read_file:
    data = json.load(read_file)
print('Data import completed.')


export_to_excel(data, cover_number)  # Export data to a spreadsheet