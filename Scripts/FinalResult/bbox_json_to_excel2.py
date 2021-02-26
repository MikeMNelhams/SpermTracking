# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:21:05 2021

@author: Shane
"""
# Import Statements
import pandas as pd
import tracking_algorithm44 as tp


def export_to_excel(data_input, tp='49', cover='00'):
    x_values = [sperm["bbox"][0] for frame in data_input["centroids"] for sperm in frame]
    y_values = [sperm["bbox"][1] for frame in data_input["centroids"] for sperm in frame]
    frame_values = [i for i, frame in enumerate(data_input["centroids"]) for _ in frame]
    w_values = [sperm["bbox"][2] for frame in data_input["centroids"] for sperm in frame]
    h_values = [sperm["bbox"][3] for frame in data_input["centroids"] for sperm in frame]

    dict_temp = {"frame": frame_values, "x_min": x_values, "y_min": y_values, "width": w_values, "height": h_values}
    df = pd.DataFrame(dict_temp)
    cover__ = "{}_{}".format(cover[0], cover[1])

    print('Writing tp {} cover {}'.format(tp, cover))
    df.to_excel(r"mojo_sperm_tracking_data_bristol\tp{}\cover{}_YOLO_NO_TRACKING_output\vid_0.xlsx".format(tp, cover__), index=False, header=False)
    print('Finished writing tp {} cover {}'.format(tp, cover))

    return 0


acceptable_tpG = ['49', '57']
acceptable_coversG = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                      '11', '12', '13', '14', '15', '16', '17', '18', '19']

cover_ = '00'
tp_ = '49'

data = tp.import_data(acceptable_tpG, cover=cover_, tp=tp_)
export_to_excel(data, cover=cover_, tp=tp_)  # Export data to a spreadsheet
