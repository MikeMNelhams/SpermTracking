import numpy as np
import sys
import pandas as pd


def _flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result


def import_ground_truth(cover='04', tp='49', verbose=False):
    """

    :param cover:
    :param tp:
    :param verbose:
    :return: np.array, list
    """
    _cover = '{}_{}'.format(cover[0], cover[1])
    pathG = r"Q:\Michael's Stuff\Eng Maths\MDM3\PhaseB- Mojo\mojo_sperm_tracking_data_bri" \
            r"stol\tp{}\cover{}_YOLO_NO_TRACKING_output\Cover{}HANDLABELLED3.csv"

    try:
        data_hl = pd.read_csv(pathG.format(tp, _cover, cover), header=None, index_col=False)
    except FileNotFoundError:
        print('File not found!')
        sys.exit(1)
    data_hl.dropna()

    data_hl_np = data_hl.to_numpy()

    if verbose:
        print('SHAPE: ', data_hl_np.shape)

    n_clusters = int(data_hl_np.shape[1] / 2)

    frame_values = []
    data_hl_np = data_hl_np.T

    clusters = [[i, []] for i in range(int(data_hl_np.shape[0] / 2))]
    x_values = []
    y_values = []

    for i in range(0, int(data_hl_np.shape[0]), 2):
        temp_column = data_hl_np[i]
        temp_column2 = data_hl_np[i+1]
        temp_column_frames = [j for j in range(len(temp_column)) if str(temp_column[j]) != 'nan']
        frame_values.append(temp_column_frames)
        temp_column = [sperm for sperm in temp_column if str(sperm) != 'nan']
        temp_column2 = [sperm for sperm in temp_column2 if str(sperm) != 'nan']
        # print('COLUMN: ', i)
        # print(temp_column)
        # print(temp_column2)
        x_values.append(temp_column)
        y_values.append(temp_column2)

        try:
            clusters[int(i / 2)][1] = [[temp_column[j], temp_column2[j], temp_column_frames[j]] for j in range(len(temp_column))]
        except ValueError:
            print('ValueError')
            if verbose:
                print('----------------------------------------------------')
                print(print(len(temp_column), len(temp_column2)), ' i: ', i)
                print(temp_column, temp_column2)
                print('----------------------------------------------------')

    x_values = _flatten_lists(x_values)
    y_values = _flatten_lists(y_values)
    frame_values = _flatten_lists(frame_values)
    X = np.column_stack((np.array(x_values), np.array(y_values), np.array(frame_values)))

    if verbose:
        print('---------------------------------------------------------')
        print('Clusters: ', clusters)
        print('x values: ', x_values)
        print('y_values: ', y_values)
        print('frame_values: ', frame_values)
        print('NUM Frames: ', len(set(frame_values)))
        print('Num points: ', len(frame_values))
        print('NUM CLUSTERS: ', n_clusters)
        print('All points as np.array: \n', X)
        print('---------------------------------------------------------')

    return X, clusters


if __name__ == '__main__':
    import_ground_truth(verbose=True)
