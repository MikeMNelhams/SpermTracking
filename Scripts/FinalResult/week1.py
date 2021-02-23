import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import sys  # For halting runtime
import itertools
import pandas as pd
# Machine Learning Imports
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan

# For working with excel
import xlsxwriter

# For saving and recovering data
import json
import pickle
import re

# For profiling the code
import cProfile
import re


# Classes
class State:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# From dataframe

class State2:
    def __init__(self, data_input, cover='00', tp='49'):
        self.x = [data_input["centroids"][i][j]["center"][0]
                  for i in range(len(data_input["centroids"]))
                  for j in range(len(data_input["centroids"][i]))]
        self.y = [data_input["centroids"][i][j]["center"][1]
                  for i in range(len(data_input["centroids"]))
                  for j in range(len(data_input["centroids"][i]))]
        self.z = [i for i in range(len(data_input["centroids"])) for _ in data_input["centroids"][i]]
        # The following a very hack line of python. I love it.
        self.X = [[sperm['center'][0], sperm["center"][1], i] if len(frame) > 0 else ''
                  for i, frame in enumerate(data_input["centroids"]) for sperm in frame]
        self.X2 = [[[sperm["center"][0], sperm["center"][1]] for sperm in frame] for frame in data_input["centroids"]]
        self.cover = cover
        self.__length = len(self.x)
        self.frame_num = len(set(self.z))
        self.tp = tp
        self.resize_factor = data_input["extra_information"]

    def __len__(self) -> int:
        return self.__length


# From State2.X np array type [[x, y, z], ...]
class State3:
    def __init__(self, X, cover='00', tp='49'):
        self.x = X[:, 0]
        self.y = X[:, 1]
        self.z = X[:, 2]
        self.X = list(X)
        self.__length = len(self.x)
        self.frame_num = len(set(self.z))  # Dunno why it has an extra, but this is somehow standard ??
        self.cover = cover
        self.tp = tp
        # Semi-fast algorithm for converting between data types. Could have preallocated list length instead
        _indices = [0]
        current_frame = 0
        for counter, frame_num in enumerate(self.z):
            if frame_num != current_frame:
                _indices.append(counter)
                current_frame = frame_num
        _indices.append(self.__length)
        _indices = np.array(_indices)
        _indices_diffs = np.diff(_indices)
        self.X2 = [[[self.x[j], self.y[j]] for j in range(int(number))]for number in _indices_diffs]

    def __len__(self)->int:
        return self.__length


# ~~~~~ Functions Library (I would make this a separate file, but cba coz we're using github and nobody links gits)
def _clamp(n, smallest, largest):
    # Clamp a value between limits. (NOT A SOFTMAX)
    return max(smallest, min(n, largest))


def _flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result


def extrapolate_missing(lists_cluster, X) -> list:
    """
    # Creates all the token sperms for all o the sperms that have not
    list_clusters: list (clusters)
    X: list [[x, y, frame], ...]
    n: int (polynomial order)
    :return: list (clusters)
    """
    token_sperms = []
    for i in range(1, len(lists_cluster)):
        if len(lists_cluster[i]) <= 300:
            x = lists_cluster[i][:, 0]
            y = lists_cluster[i][:, 1]
            z = lists_cluster[i][:, 2]
            p1_x = np.polyfit(z, x, 1)
            p1_y = np.polyfit(z, y, 1)
            z_t = np.array(range(301))
            x_t = p1_x[1] + p1_x[0] * (z_t)  # + p1_x[0]*(z_t)**2 + p1_x[0]*(z_t)**3
            y_t = p1_y[1] + p1_y[0] * (z_t)  # + p1_y[0]*(z_t)**2 + p1_y[0]*(z_t)**3
            for t in range(len(z_t)):
                token_sperms.append([x_t[t], y_t[t], z_t[t]])

    X_out = X
    # this are the extra sperms "token sperms"
    for t in range(len(token_sperms)):
        X_out.append([token_sperms[t][0], token_sperms[t][1], token_sperms[t][2]])
    return X_out


def num_points_all():
    points1 = []
    for i in range(20):
        if len(str(i)) == 1:
            points1.append(num_points('0{}'.format(str(i))))
        else:
            points1.append(num_points(str(i)))
    print(points1)
    for i in range(20):
        if len(str(i)) == 1:
            points1.append(num_points('0{}'.format(str(i)), tp='57'))
        else:
            points1.append(num_points(str(i), tp='57'))
    points1 = np.array(points1)
    print(points1)
    print(np.mean(points1))


def num_points(acceptable_tp, cover='00', tp='49', verbose=False):
    if not cover.isnumeric() or len(cover) != 2:
        print("Cover number not from 00 to 19")
        sys.exit()
    if not 0 <= int(cover) <= 19:
        print("Cover number not from 00 to 19")
        sys.exit()
    if type(tp) != str:
        print("tp not a string")
        sys.exit()
    if tp not in acceptable_tp:
        print("tp number not acceptable")
        sys.exit()
    cover_ = "{}_{}".format(cover[0], cover[1])
    if verbose:
        print('Importing video Data for cover {}'.format(cover))
    with open(pathG.format(
            tp, cover_),
              "r") as temp_read:
        data_ = json.load(temp_read)
    if verbose:
        print('Data import completed.')
    # Count the number of datapoints (array size is uneven and nested)
    x = [0
         for element_ in range(len(data_["centroids"]))
         for _ in data_["centroids"][element_]]
    return len(x)


def pickle_state(x, y, z, path="h_space00"):
    print("Pickling file {}".format(path))
    H = State(x, y, z)
    with open(path, "wb") as output_file:
        pickle.dump(H, output_file)


def unpickle_state(path="h_space00") -> "State":
    with open(path, "rb") as input_file:
        out = pickle.load(input_file)
    print("Unpickling file {}".format(path))
    return out


def num_frames(data_input):
    return len(data_input["centroids"])


def export_to_excel(data_input, filename):
    # Export the data to an xlsx spreadsheet.
    with xlsxwriter.Workbook(filename + ".xlsx") as workbook:
        worksheet = workbook.add_worksheet()

    x_values, y_values, frame_values = [], [], []

    for i in range(0, len(data_input["centroids"])):  # number of frames
        for j in range(0, len(data_input["centroids"][i])):  # number of sperms in frame
            x_values.append(data_input["centroids"][i][j]["center"][0])
            y_values.append(data_input["centroids"][i][j]["center"][1])
            frame_values.append(i)

    for i in range(len(x_values)):
        worksheet.write(i, 0, x_values[i])
        worksheet.write(i, 1, y_values[i])
        worksheet.write(i, 2, frame_values[i])

    return 0

#---------------------------------------------------------------------------------------------------------------------#
# finds the optimal polynimial number
def polynimialNumber(BTC_list):
    run = True
    r = 1
    while run:
        dif = BTC_list[r] - BTC_list[r - 1]
        if dif >= -10 or r == len(BTC_list) - 1:
            polynomial_order = r
            run = False
        r += 1
    return polynomial_order


def optimalPolynomial(sperm_number, current_cluster):
    new_lists_cluster = current_cluster[sperm_number][1]

    residual_x_list = []
    BIC_x_list = []
    residual_y_list = []
    BIC_y_list = []
    x = [item[0] for item in new_lists_cluster]
    y = [item[1] for item in new_lists_cluster]
    z = [item[2] for item in new_lists_cluster]
    result = True
    while result:
        try:
            for i in range(1, 12):
                # connect
                residual_x = np.polyfit(z, x, i, full=True)[1][0]
                residual_y = np.polyfit(z, y, i, full=True)[1][0]
                residual_x_list.append(residual_x)
                residual_y_list.append(residual_y)

                # Calculating the BIC for x
                n = len(z)  # number of data points
                k = (len(np.polyfit(z, x, i, full=True)[0]) - 1)  # Polynomial order number
                BIC = n * np.log(residual_x / n) + k * np.log(n)  # calculates the BIC of the regression
                BIC_x_list.append(BIC)
                # Calculating the BIC for y
                n = len(z)  # number of data points
                k = (len(np.polyfit(z, y, i, full=True)[0]) - 1)  # Polynomial order number
                BIC = n * np.log(residual_y / n) + k * np.log(n)  # calculates the BIC of the regression
                BIC_y_list.append(BIC)

                if i == 11:
                    result = False

        except:
            result = False
            pass


    if len(BIC_x_list) <= 1:
        polynomial_order_x = 0
    else:
        polynomial_order_x = polynimialNumber(BIC_x_list)


    if len(BIC_y_list) <= 1:
        polynomial_order_y = 0
    else:
        polynomial_order_y = polynimialNumber(BIC_y_list)

    # fits the polynomial with the most optimal polynomial order
    x = [item[0] for item in new_lists_cluster]
    y = [item[1] for item in new_lists_cluster]
    z = [item[2] for item in new_lists_cluster]
    p1_x = np.polyfit(z, x, polynomial_order_x)
    p1_y = np.polyfit(z, y, polynomial_order_y)

    new_minimum_z = min(z) - 50
    new_maximum_z = max(z) + 50
    if new_minimum_z <= 0:
        new_minimum_z = 0
    if new_maximum_z >= 300:
        new_maximum_z = 301

    z_new = np.array(range(int(new_minimum_z), int(new_maximum_z) + 1))
    poly_x = np.poly1d(p1_x)
    poly_y = np.poly1d(p1_y)


    z_t = list(set(z_new) - set(z))
    print(z_t)
    x_t = poly_x(z_t)
    y_t = poly_y(z_t)
    return (x_t, y_t, z_t)

    # finds the optimal polynomial number for x and y

#----------------------------------------------------------------------------------------------------------------------#

def calc_clusters(data_State, algorithm="kmeans", n_clusters=10, plot=True, plot_type='2d', heatmap=False,
                  min_points_for_clustering=20, return_clusters=False, write_output=True):
    valid_algorithms = ["kmeans", "dbscan", "mike", "none", "hdbscan", "gmm", "richard-dbscan", 'mike-htdbscan',
                        "richard-hdbscan", 'mike-hthdbscan', "richard+mike"]
    algorithm_t = "dbscan"  # Default algorithm
    if algorithm in valid_algorithms:
        algorithm_t = algorithm

    if data_State.frame_num == 0:
        print('TP {} cover {} has no datapoints!'.format(data_State.tp, data_State.cover))
        if plot_type == 'bar_graph':
            return 0
        sys.exit(1)

    # Find all positions of center of sperm heads for data video
    # Plots the graph by default.
    # Returns the clusters
    x_values = data_State.x.copy()
    y_values = data_State.y.copy()
    frame_values = data_State.z.copy()

    X = [[x_values[i], y_values[i], frame_values[i]] for i in range(len(x_values))]
    X = np.asarray(X)  # The data

    clusters = []  # Initialisation
    pred_y = []  # Initialisation
    n_clusters = n_clusters  # Intialisation, should change to the number of labels
    if data_State.frame_num < min_points_for_clustering:
        # Too few data points, then treat them as they are all noise
        print('TP {} cover {} has too few data points!'.format(data_State.tp, data_State.cover))
        algorithm_t = 'none'

        if plot_type == 'bar_graph':
            print('---------------------------------------------------')
            print('U: []')
            print('---------------------------------------------------')
            return 0

    elif algorithm_t.lower() == 'dbscan':
        # Calculate the predictions using dbscan
        # eps: max distance to be considered a cluster neighbour.
        # min_samples: similar to KNN, minimum neighbours to make a cluster
        model = DBSCAN(eps=12, min_samples=12, n_jobs=4)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        pred_y = model.fit_predict(X)

        # How many clusters?
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n_clusters)
        print('# of Noise clusters', n_noise)
        print('# TOTAL: ', len(pred_y))

    elif algorithm_t.lower() == 'kmeans':
        # Calculate the predictions using kmeans
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=10, random_state=0)
        pred_y = model.fit_predict(X)

    elif algorithm_t.lower() == 'mike':
        # Calculate the predictions using Mike distance tracking method
        # Precalculate the euclidean distances of each sperm from each other sperm for every frame with frame diff
        distances = calc_distances(data_State, 2, True, frame_diff=1)
        distances_f = calc_distances(data_State, 2, True, frame_diff=-1)
        # Plot a line between each sperm, using primitive tracking as the shortest distance sperm in the previous frame.
        # Add each sperm coordinate across all the frames to a separate list

        pred_y, n_clusters = track_sperms(data_State, distances, distances_f)
        algorithm_t = 'closest frame'  # Change the name to something more appropriate

    elif algorithm_t.lower() == 'richard+mike':
        # Calculate the predictions using Mike distance tracking method
        # Precalculate the euclidean distances of each sperm from each other sperm for every frame with frame diff
        distances = calc_distances(data_State, 2, True, frame_diff=1)
        distances_f = calc_distances(data_State, 2, True, frame_diff=-1)
        # Plot a line between each sperm, using primitive tracking as the shortest distance sperm in the previous frame.
        # Add each sperm coordinate across all the frames to a separate list




        pred_y, n_clusters = track_sperms(data_State, distances, distances_f)
        algorithm_t = 'closest frame'  # Change the name to something more appropriate

        # Generate the ID and a random colours for each ID
        # List comprehensions are over 100% more efficient
        clusters = [[i, []] for i in range(n_clusters)]
        # Append all of the clustering predictions to the data structure
        for i in range(len(pred_y)):
            clusters[pred_y[i]][1].append([x_values[i], y_values[i], frame_values[i]])

#---------------------------------------------------------------------------------------------------------------------#

        new_lists_cluster = clusters
        count = 0
        repeat = 0
        len_clusters = []
        lists_of_lists_cluster = new_lists_cluster
        print("len(new_lists_cluster[1])", len(new_lists_cluster[0][1]))
        old_lists_cluster = []
        # while sorted(old_lists_cluster) != sorted(lists_of_lists_cluster):
        while count < 1:
            count += 1
            count2 = 0
            old_lists_cluster = lists_of_lists_cluster
            token_sperms = []
            for speram_number in range(1, len(new_lists_cluster)):
                if len(new_lists_cluster[speram_number][1]) <= 300 and len(new_lists_cluster[speram_number][1]) >= 10:
                    count2 += 1
                    x_t, y_t, z_t = optimalPolynomial(speram_number, new_lists_cluster)
                    for t in range(0, len(z_t)):
                        token_sperms.append([x_t[t], y_t[t], z_t[t]])

            print("count2", count2)
            n_points = len(data_State)
            print("n_points", n_points)
            X = []
            for i in range(0, n_points):
                X.append([x_values[i], y_values[i], frame_values[i]])

                # this are the extra sperms "token sperms"
            for t in range(0, len(token_sperms)):
                X.append([token_sperms[t][0], token_sperms[t][1], token_sperms[t][2]])


            print("X", X)
            X = np.array(X)
            print("X[:, 2]", X[:, 2])
            X[:, 2] = [round(x) for x in X[:, 2]]
            print("INT? X[:, 2]", X[:, 2])
            print("X", X.shape[0])
            distances = calc_distances(State3(X), 2, True, frame_diff=1)
            distances_f = calc_distances(State3(X), 2, True, frame_diff=-1)

            pred_y, n_clusters = track_sperms(State3(X), distances, distances_f)
            algorithm_t = 'closest frame'  # Change the name to something more appropriate

            print("pred_y", pred_y)

            print("len(clusters)", n_clusters)

            x_values = list(X[:, 0])
            y_values = list(X[:, 1])
            frame_values = list(X[:, 2])

            print("frame_values",  frame_values)

            # fig, (ax2, ax1) = plt.subplots(1,2)
            # ax1.scatter(X[:, 0], X[:, 1], s=0.5)
            # ax2.scatter(X[:, 0], X[:, 2], s=0.5)
            #
            #
            # plt.show()
            # sys.exit()
        #
        #
        #     fig = plt.figure
        #     ax = plt.axes(projection="3d")
        #     lists_cluster = []
        #     for x in range(n_clusters):
        #         cluster_A = []
        #         for i in range(0, len(x_values)):
        #             if labels[i] == x - 1:  # this is the noise data
        #                 cluster_A.append([X[i][0], X[i][1], X[i][2]])
        #         if cluster_A != []:
        #             cluster_A = np.asarray(cluster_A)
        #             lists_cluster.append(cluster_A)
        #         r = lambda: random.randint(0, 25)
        #         random_color = '#%02X%02X%02X' % (r() * 10, r() * 10, r() * 10)
        #         if x == 0:  # The noise data print as black dots
        #             random_color = 'black'
        #         if cluster_A != []:
        #             ax.scatter(cluster_A[:, 0], cluster_A[:, 1], cluster_A[:, 2], color=random_color, s=1)
        #
        #     ax.set_xlabel('X Axes')
        #     ax.set_ylabel('Y Axes')
        #     ax.set_zlabel('Frame Axes')
        #     plt.show()
        #     sys.exit()
        #
        #
        #
        #
        #
        #     """X = []
        #     for i in range(0, len(x_values)):
        #         X.append([x_values[i], y_values[i], frame_values[i]])
        #
        #     # this are the extra sperms "token sperms"
        #     for t in range(0, len(token_sperms)):
        #         X.append([token_sperms[t][0], token_sperms[t][1], token_sperms[t][2]])
        #
        #     # Does the DBSCAN with the token sperms
        #     X = np.asarray(X)
        #     dbscan = DBSCAN(eps=10, min_samples=10)
        #     model = dbscan.fit(X)
        #     labels = model.labels_
        #
        #     # How many clusters?
        #     sample_cores = np.zeros_like(labels, dtype=bool)
        #     sample_cores[dbscan.core_sample_indices_] = True
        #     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)"""
        #
        #     """new_lists_cluster = []
        #     for x in range(n_clusters):
        #         cluster_A = []
        #         for i in range(0, len(x_values)):
        #             if labels[i] == x - 1:  # this is the noice data
        #                 cluster_A.append([x_values[i], y_values[i], frame_values[i]])
        #         if cluster_A != []:
        #             cluster_A = np.asarray(cluster_A)
        #             new_lists_cluster.append(cluster_A)"""
        #
        #     lists_of_lists_cluster = new_lists_cluster
        #     print("count:", count)
        #     print("lists_of_lists_cluster", len(lists_of_lists_cluster))
        #     len_clusters.append(len(lists_of_lists_cluster))
        #
        # print(len_clusters.index(min(len_clusters)))




    elif algorithm_t.lower() == 'hdbscan':
        # code from: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(X)
        labels = clusterer.labels_
        pred_y = clusterer.fit_predict(X)

        # How many clusters?
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n_clusters)
        print('# of Noise Clusters ', n_noise)

    elif algorithm_t.lower() == 'gmm':
        # code from: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn
        # .mixture.GaussianMixture

        # Use a DBSCAN estimate for the number of components
        model = DBSCAN(eps=13, min_samples=5, n_jobs=2)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters = math.ceil(n_clusters/2)

        # GMM clusterer
        clusterer = GaussianMixture(n_clusters, covariance_type='diag')
        clusterer.fit(X)
        pred_y = clusterer.fit_predict(X)

    elif algorithm_t.lower() == 'mike-htdbscan':
        # 1. Use Mike K-patch
        # 2. Determine mean r, mean theta for every cluster
        # 3. Use DBSCAN to group r, theta parameters
        # 4. Relabel the Mike K-patch clusters

        # 1. K-patch
        clusters = calc_clusters(data_State, algorithm='mike', plot=False, return_clusters=True)

        # 2. H-transform parameters for each cluster
        r_all_, r2_all_ = [], []
        t_all_, t2_all_ = [], []
        pred_y = [i for i in range(len(clusters))]
        for i, cluster in enumerate(clusters):
            if len(cluster[1]) > 1:
                XT = np.array(cluster[1])
                XT, XY = XT[:, [0, 2]], XT[:, [1, 2]]

                H_ = __hough_transform3(XT)
                H2_ = __hough_transform3(XY)
                r_all_.append(np.mean(H_[:, 0]))
                t_all_.append(np.mean(H_[:, 1]))
                r2_all_.append(np.mean(H2_[:, 0]))
                t2_all_.append(np.mean(H2_[:, 1]))
            else:
                pred_y[i] = -2

        pred_y = np.array(pred_y)
        noise_indices = np.where(pred_y==-2)
        noise_indices = list(noise_indices[0])
        print('# TOTAL: ', len(pred_y))

        H_ = np.column_stack((np.array(r_all_), np.array(t_all_)))
        H2_ = np.column_stack((np.array(r2_all_), np.array(t2_all_)))

        # 2.5 Plot the Graph comparing H-space to cartesian
        if plot:
            colors = [[random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), random.uniform(0.05, 0.7)] for _ in range(len(r_all_))]

            fig, (ax2, ax1) = plt.subplots(1, 2)
            fig.set_size_inches(12, 5)  # Set the sizing
            fig.suptitle('{} clustering applied to the sperm centroids for tp {} cover {}'.format(algorithm_t.upper(),
                                                                                                  data_State.tp,
                                                                                                  data_State.cover),
                         fontsize=18)
            for i, point in enumerate(H_):
                ax2.scatter(point[0], point[1], label=i, color=colors[i])

            for i, point in enumerate(H2_):
                ax1.scatter(point[0], point[1], label=i, color=colors[i])

            ax2.set_xlabel('R')
            ax2.set_ylabel(r'$\theta$')
            ax1.set_xlabel('R')
            ax1.set_ylabel(r'$\theta$')

            ax2.title.set_text(' (x, t) ')
            ax1.title.set_text(' (y, t) ')

            ax1.legend(ncol=4, loc='center left', bbox_to_anchor=(1.05, 0.45), markerscale=1, handletextpad=0.6,
                               labelspacing=0.5, columnspacing=0.5)
            fig.tight_layout()  # Rescale everything so subplots always fit
            # plt.show()

        # 3. DBSCAN the H-space points
        X = np.column_stack((H_, H2_))
        # Calculate the predictions using dbscan
        # eps: max distance to be considered a cluster neighbour.
        # min_samples: similar to KNN, minimum neighbours to make a cluster
        model = DBSCAN(eps=100, min_samples=3, n_jobs=4)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        pred_y = model.fit_predict(X)

        # How many clusters?
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) + 1
        n_noise = list(labels).count(-1)
        print('# of clusters ', n_clusters)
        print('# of Noise clusters ', n_noise)

        # 4. Convert back to our original state space
        pred_y = list(pred_y)
        for index in noise_indices:
            pred_y.insert(index, -2)
        print('# TOTAL: ', len(pred_y))

        for i in range(len(clusters)):
            clusters[i][0] = pred_y[i]
        n_clusters = len(pred_y)

        # 4.2 -1 and -2 are all diff clusters
        p = max(pred_y) + 1
        for i, pred in enumerate(pred_y):
            if pred == -1 or pred == -2:
                clusters[i][0] = p
                pred_y[i] = p
                p += 1

        print('Unique:', len(set(pred_y)))
        n_clusters = len(set(pred_y))

        clusters2 = [[i, []] for i in range(n_clusters)]

        for i, pred in enumerate(pred_y):
            clusters2[pred][1].extend(clusters[i][1])

        clusters = clusters2
        X = data_State.X

    elif algorithm_t.lower() == 'mike-hthdbscan':
        # 1. Use Mike K-patch
        # 2. Determine mean r, mean theta for every cluster
        # 3. Use DBSCAN to group r, theta parameters
        # 4. Relabel the Mike K-patch clusters

        # 1. K-patch
        clusters = calc_clusters(data_State, algorithm='mike', plot=False, return_clusters=True)

        # 2. H-transform parameters for each cluster
        r_all_, r2_all_ = [], []
        t_all_, t2_all_ = [], []
        pred_y = [i for i in range(len(clusters))]
        for i, cluster in enumerate(clusters):
            if len(cluster[1]) > 1:
                XT = np.array(cluster[1])
                XT, XY = XT[:, [0, 2]], XT[:, [1, 2]]

                H_ = __hough_transform3(XT)
                H2_ = __hough_transform3(XY)
                r_all_.append(np.mean(H_[:, 0]))
                t_all_.append(np.mean(H_[:, 1]))
                r2_all_.append(np.mean(H2_[:, 0]))
                t2_all_.append(np.mean(H2_[:, 1]))
            else:
                pred_y[i] = -2

        pred_y = np.array(pred_y)
        noise_indices = np.where(pred_y==-2)
        noise_indices = list(noise_indices[0])
        print('# TOTAL: ', len(pred_y))

        H_ = np.column_stack((np.array(r_all_), np.array(t_all_)))
        H2_ = np.column_stack((np.array(r2_all_), np.array(t2_all_)))

        # 2.5 Plot the Graph comparing H-space to cartesian
        if plot:
            colors = [[random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), random.uniform(0.05, 0.7)] for _ in range(len(r_all_))]

            fig, (ax2, ax1) = plt.subplots(1, 2)
            fig.set_size_inches(12, 5)  # Set the sizing
            fig.suptitle('{} clustering applied to the sperm centroids for tp {} cover {}'.format(algorithm_t.upper(),
                                                                                                  data_State.tp,
                                                                                                  data_State.cover),
                         fontsize=18)
            for i, point in enumerate(H_):
                ax2.scatter(point[0], point[1], label=i, color=colors[i])

            for i, point in enumerate(H2_):
                ax1.scatter(point[0], point[1], label=i, color=colors[i])

            ax2.set_xlabel('R')
            ax2.set_ylabel(r'$\theta$')
            ax1.set_xlabel('R')
            ax1.set_ylabel(r'$\theta$')

            ax2.title.set_text(' (x, t) ')
            ax1.title.set_text(' (y, t) ')

            ax1.legend(ncol=4, loc='center left', bbox_to_anchor=(1.05, 0.45), markerscale=1, handletextpad=0.6,
                               labelspacing=0.5, columnspacing=0.5)
            fig.tight_layout()  # Rescale everything so subplots always fit
            # plt.show()

        # 3. DBSCAN the H-space points
        X = np.column_stack((H_, H2_))
        # Calculate the predictions using dbscan
        # eps: max distance to be considered a cluster neighbour.
        # min_samples: similar to KNN, minimum neighbours to make a cluster
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(X)
        labels = clusterer.labels_
        pred_y = clusterer.fit_predict(X)

        # How many clusters?
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) + 1
        n_noise = list(labels).count(-1)
        print('# of clusters ', n_clusters)
        print('# of Noise clusters ', n_noise)

        # 4. Convert back to our original state space
        pred_y = list(pred_y)
        for index in noise_indices:
            pred_y.insert(index, -2)
        print('# TOTAL: ', len(pred_y))

        for i in range(len(clusters)):
            clusters[i][0] = pred_y[i]
        n_clusters = len(pred_y)

        # 4.2 -1 and -2 are all diff clusters
        p = max(pred_y) + 1
        for i, pred in enumerate(pred_y):
            if pred == -1 or pred == -2:
                clusters[i][0] = p
                pred_y[i] = p
                p += 1

        print('Unique:', len(set(pred_y)))
        n_clusters = len(set(pred_y))

        clusters2 = [[i, []] for i in range(n_clusters)]

        for i, pred in enumerate(pred_y):
            clusters2[pred][1].extend(clusters[i][1])

        clusters = clusters2
        X = data_State.X

    elif algorithm_t.lower() == 'richard-dbscan':
        # Use polynomial regression to extrapolate missing points.

        # First use initial DBSCAN
        n_beginning = len(data_State)
        dbscan = DBSCAN(eps=12, min_samples=10)
        model = dbscan.fit(X)
        labels = model.labels_

        # How many clusters?
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        lists_cluster = []
        for x in range(n_clusters):
            cluster_A = [[x_values[i], y_values[i], frame_values[i]] for i in range(len(labels)) if labels[i] == x - 1]
            cluster_A = np.asarray(cluster_A)
            lists_cluster.append(cluster_A)

        # Append the extrapolated data
        X = extrapolate_missing(lists_cluster, list(X).copy())
        X = np.asarray(X)
        print('Number of total points after extrapolation: ', len(X))

        # Perform DBSCAN again for data with missing values extrapolated
        dbscan = DBSCAN(eps=12, min_samples=10)
        model = dbscan.fit(X)
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n_clusters)
        print('# of Noise clusters', n_noise)

        # Remove the surplus points
        X = X[:n_beginning]
        x_values = X[:, 0]
        y_values = X[:, 1]
        frame_values = X[:, 2]
        labels = labels[:n_beginning]

        pred_y = labels
        n_clusters = len(set(pred_y))

        # Change the name to be more appropriate for the title
        algorithm_t = 'Linearly extrapolated 2-iter DBSCAN'

    elif algorithm_t.lower() == 'richard-hdbscan':
        # Use polynomial regression to extrapolate missing points.

        # First use initial DBSCAN
        n_beginning = len(data_State)
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(X)
        labels = clusterer.labels_

        # How many clusters?
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        lists_cluster = []
        for x in range(n_clusters):
            cluster_A = [[x_values[i], y_values[i], frame_values[i]] for i in range(len(labels)) if labels[i] == x - 1]
            cluster_A = np.asarray(cluster_A)
            lists_cluster.append(cluster_A)

        # Append the extrapolated data
        X = extrapolate_missing(lists_cluster, list(X).copy())
        X = np.asarray(X)
        print('Number of total points after extrapolation: ', len(X))

        # Perform DBSCAN again for data with missing values extrapolated
        dbscan = DBSCAN(eps=12, min_samples=10)
        model = dbscan.fit(X)
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n_clusters)
        print('# of Noise clusters', n_noise)

        # Remove the surplus points
        X = X[:n_beginning]
        x_values = X[:, 0]
        y_values = X[:, 1]
        frame_values = X[:, 2]
        labels = labels[:n_beginning]

        pred_y = labels
        n_clusters = len(set(pred_y))

        # Change the name to be more appropriate for the title
        algorithm_t = 'Linearly extrapolated 2-iter DBSCAN'

    elif algorithm_t.lower() == 'none':
        # No clustering algorithm, treat it as one cluster, with colour black
        n_clusters = len(data_State)
        pred_y = [1 for _ in data_State.x]

    if algorithm_t.lower() != 'mike-htdbscan' and algorithm_t.lower() != 'mike-hthdbscan':
        print("asdfasdfasdfasdfasdfas")
        # Generate the ID and a random colours for each ID
        # List comprehensions are over 100% more efficient
        clusters = [[i, []] for i in range(n_clusters)]

        # Append all of the clustering predictions to the data structure
        for i in range(len(pred_y)):
            clusters[pred_y[i]][1].append([x_values[i], y_values[i], frame_values[i]])

    # (Generating DISTINCT n colours is a time-complex task) Randomising uniformly is a good compromise
    colors = [[random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), random.uniform(0.05, 0.7)]
              for _ in range(n_clusters)]

    if algorithm_t.lower() == 'none':
        # Should only be black
        colors = [[0, 0, 0] for _ in range(n_clusters)]

    if algorithm_t.lower() == "linearly extrapolated 2-iter dbscan":
        # We want the noise to appear black
        colors[-1][:] = (0, 0, 0)

        # We need to swap the first and last elements since the noise is currently the last cluster rather than 1st
        colors[0], colors[-1] = colors[-1], colors[0]
        clusters[0][1], clusters[-1][1] = clusters[-1][1], clusters[0][-1]

    if heatmap:
        # https://www.google.com/url?sa=i&url=https%3A%2F%2Fmatplotlib.org%2F3.1.0%2Ftutorials%2Fcolors%2Fcolormaps
        # .html&psig=AOvVaw0reomRw7Ry3-j0zu54NIbz&ust=1613262335685000&
        # source=images&cd=vfe&ved=0CAIQjRxqFwoTCKiFpdfM5e4CFQAAAAAdAAAAABAO
        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        print([len(clusters[i][1]) for i in range(n_clusters)])
        n_frames = data_State.frame_num
        limits = [0.45, 0.6]
        # More black colours are assigned to the less complete clusters
        heat = cm.get_cmap('RdGy', n_clusters)
        colors = [heat(_clamp((n_frames - len(clusters[i][1])) / n_frames, 0, 1))
                  if not (n_frames - limits[0] * n_frames) >= len(clusters[i][1]) >= (n_frames - limits[1] * n_frames)
                  else heat(limits[1])
                  for i in range(n_clusters)]
        del heat

    # List comprehensions are over 100% more efficient
    clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n_clusters) if len(clusters[i][1]) != 0]

    if write_output:
        # Write the output to a correctly named .json file.
        _write_clusters_space_to_json(pred_y, cover=data_State.cover,
                                      tp=data_State.tp, algorithm=algorithm_t, verbose=True)

    if plot:
        if plot_type == '2d':
            fig, (ax3, ax2, ax1) = plt.subplots(1, 3)

            if n_clusters < 40:
                # Small number
                fig.set_size_inches(17, 5)  # Set the sizing
            elif n_clusters < 80:
                # Medium number
                fig.set_size_inches(18, 5)  # Set the sizing
            elif n_clusters < 150:
                # Large number
                fig.set_size_inches(19, 6)  # Set the sizing
            else:
                # Very Large number
                fig.set_size_inches(19, 7)  # Set the sizing

            fig.suptitle('{} clustering applied to the sperm centroids for tp {} cover {}'.format(algorithm_t.upper(),
                                                                                                  data_State.tp,
                                                                                                  data_State.cover),
                         fontsize=18)

            for i, cluster in enumerate(clusters_asarr):
                color_map = colors[i]
                ax1.scatter(cluster[:, 0], cluster[:, 1], label=i, color=color_map, s=5)
                ax2.scatter(cluster[:, 0], cluster[:, 2], label=i, color=color_map, s=5)
                ax3.scatter(cluster[:, 1], cluster[:, 2], label=i, color=color_map, s=5)

            ax3.set_xlabel('X Axis')
            ax3.set_ylabel('Frame Number')
            ax2.set_xlabel('Y Axis')
            ax2.set_ylabel('Frame Number')
            ax1.set_xlabel('X Axis')
            ax1.set_ylabel('Y Axis')

            # Put a legend to the right of the current axis
            if n_clusters < 40:
                # Small number
                ax1.legend(ncol=2, loc='center left', bbox_to_anchor=(1.05, 0.45), markerscale=5, handletextpad=0.6,
                           labelspacing=0.5, columnspacing=0.6)

            elif n_clusters < 60:
                # Medium number
                ax1.legend(ncol=3, loc='center left', bbox_to_anchor=(1.05, 0.45), markerscale=4.5, handletextpad=0.6,
                           labelspacing=0.5, columnspacing=0.5)
            elif n_clusters < 80:
                # Medium number
                ax1.legend(ncol=4, loc='center left', bbox_to_anchor=(1.05, 0.45), markerscale=4, handletextpad=0.6,
                           labelspacing=0.5, columnspacing=0.5)
            else:
                # Large number
                ax1.legend(ncol=5, loc='center left', bbox_to_anchor=(1.05, 0.5), markerscale=3.5, handletextpad=0.25,
                           labelspacing=0.4, columnspacing=0.3)

            ax3.title.set_text(' (A) ')
            ax2.title.set_text(' (B) ')
            ax1.title.set_text(' (C) ')

            fig.tight_layout()  # Rescale everything so subplots always fit
            fig.subplots_adjust(wspace=0.15)  # Trim the space a little between the subplots

            plt.show()
        elif plot_type == '3d':
            # This will plot the graph in 3D
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            for i, cluster in enumerate(clusters_asarr):
                color_map = colors[i]
                ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=color_map, s=1)

            ax.set_xlabel('X Axes')
            ax.set_ylabel('Y Axes')
            ax.set_zlabel('Frame number')
            plt.show()
        elif plot_type == 'bar_graph':
            fig, ax = plt.subplots()
            fig.suptitle('{} clustering totals for tp {} cover {}'.format(algorithm_t.upper(), data_State.tp,
                                                                          data_State.cover),
                         fontsize=18)
            ax.grid()  # Grids are very necessary for bar graphs
            ax.set_axisbelow(True)  # So the grid goes behind the lines
            totals = [cluster.shape[0] for cluster in clusters_asarr]
            x_ticks = range(len(totals))

            # Technically, this is deprecated, but try to stop me >:`)
            # Colouring is more based on relative distance from the number of frames
            colors = [plt.cm.viridis(abs(data_State.frame_num - n) / data_State.frame_num) for n in totals]
            for x_val, y_val, color in zip(x_ticks, totals, colors):
                ax.bar(x_val, y_val, label=data_State.cover, color=color)
            ax.set_xlabel('Cluster Number')
            ax.set_ylabel('Number of clustered centroids')
            plt.show()

    if plot_type == 'bar_graph':
        # Sneaky returning the U value where:
        # U = Mean(|n_f - x|) where x is the number of points in each cluster
        totals = [cluster.shape[0] for cluster in clusters_asarr]
        _u = [abs(data_State.frame_num - n) for n in totals]
        print('---------------------------------------------------')
        print('U: ', _u)
        print('---------------------------------------------------')
        _u = np.array(_u)
        _u = np.mean(_u) / data_State.frame_num
        print(totals)
        print(_u)
        return _u

    if return_clusters:
        return clusters

    return pred_y


def _write_clusters_space_to_json(predictions, cover='00', tp='49', algorithm='none', verbose=False):
    _old_data = import_data(acceptable_tpG, cover=cover, tp=tp)

    # Initiliase the frame structure. Massive speedups in write speeds
    data = {'centroids': [[] for _ in _old_data['centroids']]}

    # Pad the predictions so they are always 4 digit strings
    def _pad_prediction(_pred):
        pred_len = len(str(_pred))
        if pred_len == 1:
            return '000{}'.format(_pred)
        if pred_len == 2:
            return '00{}'.format(_pred)
        if pred_len == 3:
            return '0{}'.format(_pred)
        return str(_pred)

    predictions2 = [_pad_prediction(pred_) for pred_ in predictions]

    j = 0
    for i, frame in enumerate(_old_data['centroids']):
        for sperm in frame:
            data['centroids'][i].append({'bbox': sperm['bbox'],  # Keep the same BBOX (this can be changed later)
                                         'center': sperm['center'],  # Center always same
                                         'class': 1,  # Class always 1. We assume 100% precision
                                         'interpolated': False,  # Currently we throw away our interpolated points
                                         'id': predictions2[j]  # ID is equal to the cluster number
                                         })
            j += 1
    data['extra_information'] = _old_data['extra_information']

    # Determine our path from global path given. Save it in the same directory.
    cover_ = "{}_{}".format(cover[0], cover[1])
    path = pathG.format(tp, cover_)
    print('Found file: {}'.format(path))

    start = path[::-1].find('\\')

    if algorithm != 'none':
        path = path[:-start] + '{}_tracking.json'.format(algorithm)
    else:
        path = path[:-start] + 'tracking.json'

    if verbose:
        print('Writing file: {}'.format(path))

    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    if verbose:
        print('Finished writing file.')
    return 0


def __hough_transform3(data_input, origin=(0, 0)):
    # Only use this ONLY for sequenced data

    X = data_input
    if origin != (0, 0):
        X = [row - origin for row in X]

    # Slow, should use static lists rather than dynamic, but oh well
    dX = np.array([(X[1] - X[0])])
    for i in range(2, len(X)):
        dX = np.row_stack((dX, X[i] - X[i - 1]))

    X = X[1:]  # Since the first value has no distance assigned to it

    # 3. Rotate distances 90 degrees (x, y) = (-y, x)
    v = np.column_stack((-dX[:, 1], dX[:, 0]))

    # 4. Normalise each distance
    n = np.sqrt(np.power(v[:, 0], 2) + np.power(v[:, 1], 2))

    # 5. Distance = x1 dot n_hat
    # Preprocess 0s, this will be at least 50% of the data.
    indices = np.where(n == 0)
    indices = list(indices[0])  # Type conversion before iterating speeds up
    # To stop division by 0, we want to have r = 0 and theta = 0, these values will be removed LATER, but kept for now
    n[indices] = 1
    v[indices] = 0

    n_hat = v / n[:, None]  # Works

    # 6. Equivalent to a matrix dot product
    r = np.sum(X[:, :] * n_hat, axis=1)
    theta = np.arctan2(v[:, 1], v[:, 0]) * (180 / math.pi)
    H = np.column_stack((r, theta))

    return H


# data_input, power, root=True. Calculate the distance for given cover data. frame_diff is the difference between frames
def calc_distances(data_State, power=2, root=True, frame_diff=1):
    # Calculate the distance between each and every sperm for every frame, done for distance between 'frame_diff' frames
    p = float(power)

    # Overall, the relevant data is in form: data["centroids"][frame_number][sperm_number]["desired_info_on_sperm"]
    # len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
    # [FRAME 1:[[coord 1x, coord 1y], [coord 2x, coord2y], etc], FRAME2: [[],[], ...], ..., FRAME 301: ...]

    X = data_State.X2

    loop_range = range(frame_diff, len(X))  # This will calculate distance between current and future
    if frame_diff < 0:
        # This is calculating distance between current and previous
        loop_range = range(0, len(X) + frame_diff)

    # Compute the distances for each frame between every single sperm
    # sperm[0] is the x coord, sperm[1] is the y coord
    # Work out the distance measure for power p for each sperm wrt to each other sperm.
    # Distance to itself should be always be 0 if frame_diff is 0.
    # Not yet coded for manhattan distance, power = 1
    # The 0s mark new sperms currently
    dist = []  # Just for localisation initialisation
    if root:
        dist = [[[np.power((sperm[0] - sperm2[0]) ** p + (sperm[1] - sperm2[1]) ** p, 1 / p)
                  for sperm2 in X[frame_i - frame_diff]]
                 for sperm in X[frame_i]]
                for frame_i in loop_range]
    else:
        if power == 1:
            dist = [[[np.abs(sperm[0] - sperm2[0]) + np.abs(sperm[1] - sperm2[1])
                      for sperm2 in X[frame_i - frame_diff]]
                     for sperm in X[frame_i]]
                    for frame_i in loop_range]
        else:
            dist = [[[(sperm[0] - sperm2[0]) ** p + (sperm[1] - sperm2[1]) ** p
                      for sperm2 in X[frame_i - frame_diff]]
                     for sperm in X[frame_i]]
                    for frame_i in loop_range]

    # Add 0s since the first frame_diff distances are technically between itself
    if frame_diff >= 0:
        for i in range(frame_diff):
            dist.insert(i, [[0 for _ in X[i]] for _ in X[i]])
    else:
        for i in range(np.abs(frame_diff)):
            dist.insert(len(X) - 1 + frame_diff, [[0 for _ in X[len(X) - 1 + frame_diff]]
                                                  for _ in X[len(X) - 1 + frame_diff]])
    return dist


def get_distance(data_State, frame_n, i, j, frame_diff=1, distance_input=None, power=2, root=True):
    """
    This is NOT used, however, it is nice for testing the distance moved between frames for a sperm
    :param data_State: object with all sperm data
    :param frame_n: number of frames (all can be found with data_State.num_frame
    :param i: sperm i frame num
    :param j: sperm j frame num
    :param frame_diff: difference between the frame
    :param distance_input: whether the distances array is precalced
    :param power: int, euclidean distance is 2.
    :param root: Whether to perform the root for euclidean distance
    :return: float
    """
    # Return the distance between sperm i and sperm j in frame_n
    dist = distance_input
    if distance_input is None:
        dist = calc_distances(data_State, power=power, root=root, frame_diff=frame_diff)
    elif frame_n < frame_diff:
        print('Error, frame diff for distance was larger than frame_n')
        sys.exit()

    output = dist[frame_n][i][j]
    # REMEMBER THAT SOMETIMES NEW SPERMS ENTER THE FRAME, so diff might not make sense.
    # I Need to include IDs and searches instead of indexes for a perfect algorithm.
    # Issue is that this will massively slow down the algorithm. The data needs to instead be processed
    return output


def correct_for_missing(smallest_indices, missing_indices):
    predictions = []
    counter = 0
    for index in smallest_indices:
        temp = index
        while (temp + counter) in missing_indices:
            counter += 1
        predictions.append(temp + counter)
    return predictions


# Returns pred_y, n_max
def track_sperms(data_State, distance_input=None, distance_f_input=None, power=2, root=True, frame_diff=1):
    # Generate the distances if not done already
    dist = distance_input
    if distance_input is None:
        dist = calc_distances(data_State, power, root, frame_diff=frame_diff)
    dist_f = distance_f_input
    if distance_f_input is None:
        dist_f = calc_distances(data_State, power, root, frame_diff=-frame_diff)

    # Return error if frame_diff is negative
    if frame_diff < 1:
        print('Error, frame_diff is negative for sperm tracking')
        sys.exit(1)

    # Case 1:
    # _________
    # Sperm number stays same. --> Link each sperm by closest distance to previous sperm.
    # Mapping is same as previous method.
    # Case 2:
    # _________
    # Sperm number has decreased --> Link each sperm by closest distance to previous sperm.
    # Identify the sperm that has left the frame.
    # Each sperm ID will need correcting to adjust for the missing sperms.
    # Case 3:
    # _________
    # Sperm number has increased --> Find each sperm by closest distance. Many to One.
    # Only link the smallest dist sperm, the others will need NEW bins. No reusing old bins
    # Add the first frame of sperm 'predictions' as themselves
    pred_y = [list(range(len(data_State.X2[i]))) for i in range(frame_diff)]
    previous_n = len(data_State.X2[frame_diff-1])  # Previous n for comparison
    starting_len = len(data_State.X2[frame_diff-1])  # The starting length
    missing_indices = []
    new_sperms = []
    for frame_i, frame in enumerate(data_State.X2[frame_diff:], frame_diff):
        # Step 1. Find closest previous sperms
        # Step 2. Decide which case it is. Described above.
        # Step 3. Make predictions array
        # Step 4. Append the sperm coords to the corresponding bin prediction
        smallest_indices = []
        n = len(frame)
        duplicates = []
        for sperm_i in range(len(frame)):
            # Smallest value that isn't already a valid ID
            smallest_index = np.argmin(dist[frame_i][sperm_i])  # Index of the nearest previous sperm
            if smallest_index not in smallest_indices:
                smallest_indices.append(smallest_index)
            else:
                duplicates.append(smallest_indices)

        for i in range(len(duplicates)):
            smallest_indices.append(starting_len+i)

        smallest_indices = correct_for_missing(smallest_indices, missing_indices)

        # print('frame#: ', frame_i, ' N: ', n, ' Shown: ', smallest_indices)
        # print([frame[sperm_i]["center"] for sperm_i in range(len(frame))])

        # Case 1. Predictions stays the same
        if n == previous_n:
            pass

        # Case 2
        elif n > previous_n:
            smallest_dist = []
            for sperm_j in range(len(data_State.X2[frame_i])):
                smallest_dist.append(dist[frame_i][sperm_j][int(np.argmin(dist[frame_i][sperm_j]))])

            k = n - previous_n  # The number of sperms that left the frame
            largest_minima = np.argpartition(smallest_dist, -k)[-k:]
            largest_minima = [smallest_indices[index] for index in largest_minima]

            # Add the new sperms to the new sperm list
            for index in largest_minima:
                new_sperms.append(index)

        # Case 3:
        else:
            # Find the missing sperms point by looking for the k largest minimum distances in the PREVIOUS mapping.
            # Calculate the largest of the smallest forwards distances for the previous frame
            smallest_fdist = []
            for sperm_j in range(len(data_State.X2[frame_i - frame_diff])):
                smallest_fdist.append(dist_f[frame_i - frame_diff][sperm_j]
                                      [int(np.argmin(dist_f[frame_i - frame_diff][sperm_j]))])

            k = previous_n - n  # The number of sperms that left the frame
            largest_minima = np.argpartition(smallest_fdist, -k)[-k:]
            largest_minima = [pred_y[frame_i-frame_diff][index] for index in largest_minima]

            # Locate the missing sperms:
            for index in largest_minima:
                missing_indices.append(index)

            smallest_indices = []
            duplicates = []
            for sperm_i in range(len(frame)):
                # Smallest value that isn't already a valid ID
                smallest_index = np.argmin(dist[frame_i][sperm_i])  # Index of the nearest previous sperm
                if smallest_index not in smallest_indices:
                    smallest_indices.append(smallest_index)
                else:
                    duplicates.append(smallest_indices)

            for i in range(len(duplicates)):
                smallest_indices.append(starting_len + i)

            smallest_indices = correct_for_missing(smallest_indices, missing_indices)

        pred_y.append(smallest_indices)
        # print('Previous: ', pred_y[frame_i - frame_diff])
        # print('Prediction: ', smallest_indices)
        # print('-------')
        previous_n = n

    # Assume that the n bins are unique to each sperm
    # n_max is the maximum number of bins required, since often new sperms enter a frame
    # Flatten pred_y
    pred_y = _flatten_lists(pred_y)
    n_clusters_out = max(pred_y) + 1
    # print(len(pred_y), ' THE SHOULD BE SAME ', len(data_State))
    return pred_y, n_clusters_out


def import_data(acceptable_tp, cover='00', tp='49', verbose=False):
    if not cover.isnumeric() or len(cover) != 2:
        print("Cover number is not between 00 and 19")
        sys.exit(1)
    if not 0 <= int(cover) <= 19:
        print("Cover number not from 00 to 19")
        sys.exit(1)
    if type(tp) != str:
        print("tp not a string")
        sys.exit(1)
    if tp not in acceptable_tp:
        print("tp number not acceptable")
        sys.exit(1)
    cover_ = "{}_{}".format(cover[0], cover[1])
    if verbose:
        print('Importing video Data for tp {} cover {}'.format(tp, cover))
    with open(pathG.format(
            tp, cover_),
            "r") as temp_read:
        data_ = json.load(temp_read)
    if verbose:
        print('Data import completed.')
    return data_


def produce_histogram(cover='00', tp='49', verbose=False, draw_type='bar', bin_count=25, outlierline=False, plot=True,
                      cutoff=-1):
    """
    :param cutoff: float (maximum allowed distance a sperm can travel)
    :param outlierline: bool
    :param plot: bool
    :param bin_count: int (num of bins)
    :param draw_type: str (hisstype)
    :param cover: cover number
    :param tp: tp number
    :param verbose: bool, debug
    :return: float RMSA
    """
    draw_type_t = draw_type
    if draw_type not in acceptable_draw_typesG:
        draw_type_t = 'bar'

    # Alot of cleverly optimised trickery to plot a fast histogram ~ O(m*n)
    data = import_data(acceptable_tpG, cover=cover, tp=tp, verbose=verbose)
    data = State2(data, cover=cover, tp=tp)
    total = len(data)
    distances = calc_distances(data)
    distances = _flatten_lists(distances)

    # Make sure none of the distances are empty. Do this with an except clause
    try:
        # Closest distances for each sperm
        distances = [min(_list) for _list in distances]
    except ValueError:
        if verbose:
            print("Distances Error")
        return -1

    # distances = _flatten_lists(_flatten_lists(distances))
    # Remove the initial superficial zeroes. Theoretically, you can apply demorgan's and speed this up with list comp
    distances_temp = []
    flag = 0 if distances[0] == 0 else 1
    for counter, element in enumerate(distances):
        if not flag:
            if element != 0:
                flag = 1
        else:
            distances_temp.append(element)
    distances = distances_temp

    # Recalculate the bins TWICE, to remove bins containing 1 data point, so we can ignore outliers
    # The outliers ARE outliers,
    # because they are generated when a sperm leaves frame and a new one enters within the SAME frame
    # Flawless trimming of outliers by purely observing the data is provably impossible.
    # My suggestion would be to improve the AI used to detect sperms
    # Alternative approach:
    # https://stackoverflow.com/questions/51329109/histogram-hide-empty-bins/51333497
    _, bins, _ = plt.hist(distances, bin_count, alpha=0.75, density=True)
    plt.clf()
    plt.close()
    cutoff_t = -1
    if cutoff == -2:
        cutoff_t = max(distances) + 1
    elif cutoff != -1:
        cutoff_t = cutoff
    else:
        cutoff_t = bins[5]

    if verbose:
        print('Distances: ', distances)
        print('Cutoff: ', cutoff_t)
    # Calculate the Root Mean Squared Average
    distances = np.array(distances)
    distances = distances[distances <= cutoff_t]
    rmsa = np.sqrt(np.mean(distances ** 2))

    if plot:
        plt.figure(figsize=(7.8, 5))  # Set the sizing
        distances = distances_temp
        n, bins, patches = plt.hist(distances, bin_count, range=(0, cutoff_t), alpha=0.75, histtype=draw_type_t)

        if verbose:
            print(distances)
            print(bins)

        # Technically, this is deprecated, but try to stop me >:`)
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.viridis(n[i] / max(n)))

        if outlierline:
            # Plot a dotted red line indicating data to ignore
            xlims = plt.gca().get_xlim()  # X max
            plt.hlines(0.001*total, xlims[0], xlims[1], colors='red', linestyles='dashed')

        plt.title('Histogram for the distances travelled by sperms between frames for tp {} and cover {}'.format(tp,
                                                                                                                 cover))
        plt.xlabel('Distance travelled in 1 frame')
        plt.ylabel('Frequency (Total {})'.format(total))
        plt.grid()
        plt.rc('axes', axisbelow=True)  # So the grid goes behind the lines
        plt.show()
    return rmsa


def evaluate_U_success(algorithm='dbscan', tps=('49', '57'), verbose=False, plot=True, max_u_value=-1.0,
                       mean_line=True, bar_width=0.35, bar_gap_scale=1.0) -> list:
    """
    Compare the ability of an algorithm across a tp for all covers
    :param algorithm: str
    :param tps: list or tuple
    :param verbose: bool
    :param plot: bool
    :param max_u_value: float (-1 is equivelent to none)
    :param mean_line: bool
    :param bar_width: float (+ve)
    :param bar_gap: float (+ve)
    :return: list of U values
    """
    # Check the TP are all accounted for
    for _tp in tps:
        if _tp not in acceptable_tpG:
            print("Unknown test patient {}".format(_tp))
            print("Currently accepts {}".format(acceptable_tpG))

    # Preallocate array structure
    u_all = [[] for _ in range(len(tps))]

    # Number of covers
    p = len(acceptable_coversG)

    for i, _tp in enumerate(tps):
        if verbose:
            print('Calculating U for {} covers'.format(p))
            for j, cover in enumerate(acceptable_coversG):
                print('Done Cover {} of {}'.format(j+1, p))
                data = import_data(acceptable_tpG, cover=cover, tp=_tp, verbose=verbose)
                data = State2(data, cover=cover, tp=_tp)
                u_all[i].append(calc_clusters(data, plot=False, algorithm=algorithm, plot_type='bar_graph'))
        else:
            for cover in acceptable_coversG:
                data = import_data(acceptable_tpG, cover=cover, tp=_tp, verbose=verbose)
                data = State2(data, cover=cover, tp=_tp)
                u_all[i].append(calc_clusters(data, plot=False, algorithm=algorithm, plot_type='bar_graph'))

    mean = np.mean(_flatten_lists(u_all))
    standard_deviation = np.std(_flatten_lists(u_all))

    if plot:
        # Maximum graph point should scale the max y axis and the colour scale should be distributed to match
        maximum = max(u_all)
        if max_u_value != -1:
            maximum = max_u_value

        # Generate the colour maps.
        viridis = cm.get_cmap('viridis', p)
        colors = [[viridis(u / maximum) for u in _tp] for _tp in u_all]

        # Plot the data
        fig, ax = plt.subplots()
        fig.suptitle('Evaluating {} for test patients {}'.format(algorithm.upper(), ', '.join(tps)),
                     fontsize=18)

        ax.grid()  # Grids are very necessary for bar graphs
        ax.set_axisbelow(True)  # So the grid goes behind the lines

        labels = acceptable_coversG  # The x-tick labels
        x = np.arange(len(labels)) * bar_gap_scale  # The label locations
        n_tp = len(tps)  # Number of different TPs

        rects = []  # For editing the bars

        if n_tp == 1:
            # No bias for singular bars
            width_biases = [0 for _ in range(n_tp)]

        elif n_tp % 2 == 0:
            # Some next level index math list comprehension. Bask in my greatness mortal.
            width_biases = [- (1 / 2) + j * (1 / n_tp) if j < n_tp / 2 else ((1 / n_tp) * (j - n_tp / 2 + 1))
                            for j in range(n_tp)]
            width_biases = [width_bias * bar_width for width_bias in width_biases]
        else:
            print('ODD NUMBER TP SIZE {} UNSUPPORTED CURRENTLY'.format(n_tp))
            sys.exit(1)

        # Plot the bars
        for _tp in tps:
            if verbose:
                print('Plotting tp {}...'.format(_tp))

            for width_bias, y_vals, color_val in zip(width_biases, u_all, colors):
                rect = ax.bar(x + width_bias, y_vals, bar_width, label=_tp, color=color_val, edgecolor='black')
                rects.append(rect)

        # Display a red dotted total mean
        if mean_line:
            xlims = plt.gca().get_xlim()  # X max
            plt.hlines(mean, xlims[0], xlims[1], colors='red', linestyles='dashed', label='Mean', linewidth=2)
            ax.legend(['Mean'])
            if verbose:
                print('Mean: ', mean)
                print('Standard Deviation: ', standard_deviation)

        # X ticks properly
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        # Limit the graph to the maximum plus a little bit of space
        axes = plt.gca()
        axes.set_ylim([0, maximum+0.05])

        ax.set_xlabel('Cover Number')
        ax.set_ylabel('U value')
        plt.show()

    return mean


# ~~~~~ Main ~~~~~
# We could pull the data all from a github using pygithub and PAT keys/SSH keys
# However, I am unsure if their data is copyrighted
# In which case we would need a license to upload their data to github. So this will do.
# len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
def run_main(tp='49', cover='00', plot=False, algorithm='dbscan', verbose=False, plot_type='2d', heatmap=False):

    data = import_data(acceptable_tpG, cover=cover, tp=tp, verbose=verbose)

    # print(data["extra_information"])  # Resize factor for the data
    # print(num_frames(data))  # Calc the number of frames for the data. Might not always be 301
    # total__ = num_points(acceptable_tpG, cover=cover, tp=tp)
    # print(total__)
    data = State2(data, cover=cover, tp=tp)

    # Cluster and plot the clusters for their 2D projections
    calc_clusters(data, plot=plot, algorithm=algorithm, plot_type=plot_type, heatmap=heatmap, write_output=True)


# Globals
acceptable_tpG = ['49', '57']  # Acceptable TP covers
acceptable_coversG = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                      '11', '12', '13', '14', '15', '16', '17', '18', '19']
acceptable_draw_typesG = ['bar', 'step', 'stepfilled']  # 2021/02 These are the only useful ones we want

# Generic Path. Obviously change your path to the directory with the data if this code returns errors.
pathG = r"mojo_sperm_tracking_data_bristol\tp{}\cover{}_YOLO_NO_TRACKING_output\centroids_with_meta.json"

min_points_for_cluster = 20  # Anything less will be treated as no clustering
fps = 60  # The frames per second for the all the data is 60 per second
mu = 0.13  # The pixel pitch in microns (As given by Mojo) Doesn't change any calculations

# export_to_excel(data, "MDM3_MOJO2")  # Export data to a spreadsheet
# produce_histogram(draw_type='bar', bin_count=100, cover='00', cutoff=-1, plot=True)

"""
evaluate_U_success('kmeans', tps=['49', '57'], verbose=True, max_u_value=3, mean_line=True,
                   bar_width=0.4, bar_gap_scale=1)
"""

run_main(algorithm='richard+mike', cover='00', plot=True)