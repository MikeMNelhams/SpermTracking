# Import Statements
import json
import math
import numpy as np
from scipy.stats.stats import pearsonr
import random
import matplotlib.pyplot as plt
import sys  # For halting runtime
from mpl_toolkits.mplot3d import axes3d

# Machine Learning Imports
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan

from sklearn import mixture

# For working with excel
import xlsxwriter

# For saving and recovering data
import pickle

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
        self.X = [[self.x[i], self.y[i], self.z[i]]
                  for i in range(len(data_input["centroids"]))
                  for _ in range(len(data_input["centroids"][i]))]
        self.X2 = [[[sperm["center"][0], sperm["center"][1]] for sperm in frame] for frame in data_input["centroids"]]
        self.cover = cover
        self.__length = len(self.x)
        self.__frame_num = len(set(self.z))
        self.tp = tp

    def __len__(self)->int:
        return self.__length


# From State2.X np array type [[x, y, z], ...]
class State3:
    def __init__(self, X, cover='00', tp='49'):
        self.x = X[:, 0]
        self.y = X[:, 1]
        self.z = X[:, 2]
        self.X = list(X)
        self.__length = len(self.x)
        self.__frame_num = len(set(self.z))  # Dunno why it has an extra, but this is somehow standard ??
        self.cover = cover
        self.tp = tp
        print('Number of frames: ', self.__frame_num)
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
        print("Indices: ", self.X2)

    def __len__(self)->int:
        return self.__length


# ~~~~~ Functions Library (I would make this a separate file, but cba coz we're using github and nobody links gits)
def _flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result


def extrapolate_missing(lists_cluster, X, n=1)->list:
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


def standard_deviation(x):
    return np.std(x)


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
    with open(r"mojo_sperm_tracking_data_bristol\tp{}\cover{}_YOLO_NO_TRACKING_output\centroids_with_meta.json".format(
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


def unpickle_state(path="h_space00")->"State":
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


def calc_clusters(data_State, algorithm="kmeans", n_clusters=10, plot=True, plot_type='2d'):
    valid_algorithms = ["kmeans", "dbscan", "mike", "none", "hdbscan", "gmm", "htdbscan", "richard"]
    algorithm_t = "dbscan"  # Default algorithm
    if algorithm in valid_algorithms:
        algorithm_t = algorithm

    # Find all positions of center of sperm heads for data video
    # Plots the graph by default.
    # Returns the clusters
    x_values = data_State.x.copy()
    y_values = data_State.y.copy()
    frame_values = data_State.z.copy()

    X = [[x_values[i], y_values[i], frame_values[i]] for i in range(len(x_values))]
    X = np.asarray(X)  # The data

    pred_y = []  # Initialisation
    n = 0  # Intialisation, should change to the number of labels

    if algorithm_t.lower() == 'dbscan':
        # Calculate the predictions using dbscan
        # eps: max distance to be considered a cluster neighbour.
        # min_samples: similar to KNN, minimum neighbours to make a cluster
        model = DBSCAN(eps=12, min_samples=10, n_jobs=4)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        pred_y = model.fit_predict(X)

        # How many clusters?
        n = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n)
        print('# of Noise clusters', n_noise)

    elif algorithm_t.lower() == 'kmeans':
        # Calculate the predictions using kmeans
        n = n_clusters  # Number of clusters
        model = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=10, random_state=0)
        pred_y = model.fit_predict(X)

    # Calculate the predictions using Mike distance tracking method
    elif algorithm_t.lower() == 'mike':

        # Precalculate the euclidean distances of each sperm from each other sperm for every frame with frame diff
        # linear_velocities(data, fps, mu, distance_input=distances_arr)  # Unfinished.
        distances = calc_distances(data_State, 2, True, frame_diff=1)
        distances_f = calc_distances(data_State, 2, True, frame_diff=-1)
        # Plot a line between each sperm, using primitive tracking as the shortest distance sperm in the previous frame.
        # Add each sperm coordinate across all the frames to a separate list
        # track_sperms(data, power=2, plot=True)

        pred_y, n = track_sperms(data_State, distances, distances_f)
        del pred_y[-1]  # Remove the last row of predictions, can't remember why I do this
        algorithm_t = 'closest frame'  # Change the name to something more appropriate
    elif algorithm_t.lower() == 'hdbscan':
        # code from: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(X)
        labels = clusterer.labels_
        pred_y = clusterer.fit_predict(X)

        # How many clusters?
        n = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n)
        print('# of Noise Clusters ', n_noise)

    elif algorithm_t.lower() == 'gmm':
        # code from: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn
        # .mixture.GaussianMixture

        # Use a DBSCAN estimate for the number of components
        model = DBSCAN(eps=13, min_samples=5, n_jobs=2)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        pred_y = model.fit_predict(X)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        n = math.ceil(n/2)

        # GMM clusterer
        clusterer = GaussianMixture(n, covariance_type='diag')
        clusterer.fit(X)
        pred_y = clusterer.fit_predict(X)

    elif algorithm_t.lower() == 'htdbscan':
        p1, p2, p3 = calc_hough_transform(data_State, plot=False)
        print("Clustering H-space...")
        """
        # Remove all the rows where -r_tol < r < r_tol
        r_tol = input("Distance tolerance? ")
        if not r_tol.isnumeric():
            r_tol = float(r_tol)
            if r_tol <= 0:
                print("Input tolerance was not a positive non-zero float")
                sys.exit()
        """
        r_tol = 5
        indices = np.where(p1 > float(r_tol))
        indices = list(indices[0])

        p1 = p1[indices]
        p2 = p2[indices]
        p3 = p3[indices]

        indices = np.where(p2 > float(r_tol))
        indices = list(indices[0])
        p1 = p1[indices]
        p2 = p2[indices]
        p3 = p3[indices]*100

        # Select and stack
        q = 100000
        p1 = np.column_stack((p1[:q], p2[:q], p3[:q]))  # This operation is slow, not as slow as clustering lol
        print("Done stacking")

        # Cluster the model
        model = DBSCAN(eps=30, min_samples=4, n_jobs=4)
        pred_y = model.fit_predict(p1)
        labels = model.labels_

        # How many clusters?
        n = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n)
        print('# of Noise Clusters ', n_noise)
        # print('LABELS: ', labels)  # The labels for every point

        # Generate the ID and a random colours for each ID
        # List comprehensions are over 100% more efficient
        clusters = [[i, []] for i in range(n)]

        # Could make unique colours for each cluster, but randomising is a good compromise
        # (Generating DISTINCT n colours is a time-complex task)
        colors = [[random.uniform(0, 0.7), random.uniform(0, 0.7), random.uniform(0, 0.7)] for _ in range(n)]

        # Append all of the clustering predictions to the data structure
        for i in range(len(pred_y)):
            clusters[pred_y[i]][1].append([p1[i, 0], p1[i, 1], p1[i, 2]])

        # List comprehensions are over 100% more efficient
        clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n)]

        if plot:
            fig, (ax3, ax2, ax1) = plt.subplots(1, 3)
            fig.set_size_inches(20, 5)  # Set the sizing
            fig.suptitle('{} clustering applied to the sperm centroids for cover 00'.format(algorithm_t.upper()),
                         fontsize=18)
            for i, cluster in enumerate(clusters_asarr):
                color_map = colors[i]
                ax1.scatter(cluster[:, 0], cluster[:, 1], label=i, color=color_map, s=5)
                ax2.scatter(cluster[:, 0], cluster[:, 2], label=i, color=color_map, s=5)
                ax3.scatter(cluster[:, 1], cluster[:, 2], label=i, color=color_map, s=5)

            ax3.set_xlabel('X Axis')
            ax3.set_ylabel('Angle')
            ax2.set_xlabel('Y Axis')
            ax2.set_ylabel('Angle')
            ax1.set_xlabel('X Axis')
            ax1.set_ylabel('Y Axis')

            # Put a legend to the right of the current axis
            ax1.legend(ncol=3, loc='center left', bbox_to_anchor=(1, 0.45), markerscale=5, handletextpad=0.6,
                       labelspacing=0.5, columnspacing=0.6)

            ax3.title.set_text(' (A) ')
            ax2.title.set_text(' (B) ')
            ax1.title.set_text(' (C) ')
            plt.show()
        return [clusters_asarr]

    elif algorithm_t.lower() == 'richard':
        # Use polynomial regression to extrapolate missing points.
        r = lambda: random.randint(0, 25)
        dbscan = DBSCAN(eps=12, min_samples=10)
        model = dbscan.fit(X)
        labels = model.labels_

        # How many clusters?
        n = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print('# of clusters ', n)
        print(labels)
        fig = plt.figure
        ax = plt.axes(projection="3d")
        lists_cluster = []
        for x in range(n):
            cluster_A = [[x_values[i], y_values[i], frame_values[i]] for i in range(len(labels)) if labels[i] == x - 1]
            cluster_A = np.asarray(cluster_A)
            lists_cluster.append(cluster_A)
            random_color = '#%02X%02X%02X' % (r() * 10, r() * 10, r() * 10)
            if x == 0:  # The noise data print as black dots
                random_color = 'black'
            ax.scatter(cluster_A[:, 0], cluster_A[:, 1], cluster_A[:, 2], color=random_color, s=1)

        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Frame Axes')
        plt.show()

        print('Number of total points ', len(X))
        X = extrapolate_missing(lists_cluster, list(X).copy())  # Doesn't work as it should
        X = np.asarray(X)
        myState = State3(X, cover=data_State.cover, tp=data_State.tp)
        print('Number of total points ', len(X))
        calc_clusters(myState, algorithm='mike', plot=plot)

        # How many clusters?
        dbscan = DBSCAN(eps=12, min_samples=10)
        model = dbscan.fit(X)
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('# of clusters ', n_clusters)
        print('# of points ', len(labels))

        ax = plt.axes(projection="3d")
        plt.title("Middle bit")
        lists_cluster = []
        for x in range(n_clusters):
            cluster_A = [[X[i][0], X[i][1], X[i][2]] for i in range(len(labels)) if labels[i] == x - 1]
            cluster_A = np.asarray(cluster_A)
            lists_cluster.append(cluster_A)
            random_color = '#%02X%02X%02X' % (r() * 10, r() * 10, r() * 10)
            if x == 0:  # The noise data print as black dots
                random_color = 'black'
            ax.scatter(cluster_A[:, 0], cluster_A[:, 1], cluster_A[:, 2], color=random_color, s=1)

        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Frame Axes')
        plt.show()

        fig = plt.figure
        ax = plt.axes(projection="3d")
        lists_cluster = []
        for x in range(n_clusters):
            cluster_A = [[X[i][0], X[i][1], X[i][2]] for i in range(len(labels)) if labels[i] == x - 1]
            cluster_A = np.asarray(cluster_A)
            lists_cluster.append(cluster_A)
            random_color = '#%02X%02X%02X' % (r() * 10, r() * 10, r() * 10)
            if x == 0:  # The noise data print as black dots
                random_color = 'black'
            ax.scatter(cluster_A[:, 0], cluster_A[:, 1], cluster_A[:, 2], color=random_color, s=1)

        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Frame Axes')
        plt.show()

        return X, labels

    elif algorithm_t.lower() == 'none':
        # No clustering algorithm, treat it as one cluster, with colour black
        n = len(data_State)
        pred_y = [1 for _ in data_State.x]

    # Generate the ID and a random colours for each ID
    # List comprehensions are over 100% more efficient
    clusters = [[i, []] for i in range(n)]

    # (Generating DISTINCT n colours is a time-complex task) Randomising uniformly is a good compromise
    colors = [[random.uniform(0, 0.7), random.uniform(0, 0.7), random.uniform(0, 0.7)] for _ in range(n)]

    if algorithm_t.lower() == 'none':
        # Should only be black
        colors = [[0, 0, 0] for _ in range(n)]

    # Append all of the clustering predictions to the data structure
    for i in range(len(pred_y)):
        clusters[pred_y[i]][1].append([x_values[i], y_values[i], frame_values[i]])

    # List comprehensions are over 100% more efficient
    clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n) if len(clusters[i][1]) != 0]

    if plot:
        if plot_type == '2d':
            fig, (ax3, ax2, ax1) = plt.subplots(1, 3)

            if n < 40:
                # Small number
                fig.set_size_inches(17, 5)  # Set the sizing
            elif n < 80:
                # Medium number
                fig.set_size_inches(18, 5)  # Set the sizing
            elif n < 150:
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

            if algorithm.lower() == 'kmeans':
                # Only kmeans provides cluster centroids
                ax1.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=80,
                            color=colors)
                ax2.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 2], s=80,
                            color=colors)
                ax3.scatter(model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], s=80,
                            color=colors)

            ax3.set_xlabel('X Axis')
            ax3.set_ylabel('Frame Number')
            ax2.set_xlabel('Y Axis')
            ax2.set_ylabel('Frame Number')
            ax1.set_xlabel('X Axis')
            ax1.set_ylabel('Y Axis')

            # Put a legend to the right of the current axis
            if n < 50:
                # Small number
                ax1.legend(ncol=2, loc='center left', bbox_to_anchor=(1.05, 0.45), markerscale=5, handletextpad=0.6,
                           labelspacing=0.5, columnspacing=0.6)
            elif n < 80:
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

            if algorithm.lower() == 'kmeans':
                # Only kmeans uses cluster centroids
                ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], s=100,
                           color=colors)

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
            bars1 = ax.bar(x_ticks, totals, label='Clusters', color='black')
            ax.set_xlabel('Cluster Number')
            ax.set_ylabel('Number of clustered centroids')
            plt.show()

    return 0


def __hough_transform2(data_input, origin=(0, 0), sequence=True):
    """

    :param data_input: 2D np-array
    :param origin: 2D vector
    :param sequence: Is the data a sequence of points?
    :return: 2D np-array state space
    """
    X = data_input
    print(np.shape(X))
    # 1. Calc distances to origin
    if origin != (0, 0):
        X = [row - origin for row in X]

    # 2. Distance to previous point
    ids = []  # Initialisation
    if sequence:
        # Sequence of points reduces complexity from n^2 to n
        dX = np.array([(X[1] - X[0])])
        for i in range(2, len(X)):
            dX = np.row_stack((dX, X[i] - X[i - 1]))
        ids = [i for i in range(len(X) - 1)]
    else:
        # Randomly spaced data points which do not form a sequence
        print('Generating Distance data...')
        m = len(X)
        # Faster method O((n^2 - n)/2) compared to O(n^2), but I am not generating IDS for this, fuck that
        # dX = [X[j] - X[i] for i in range(m) for j in range(i+1, m)]

        dX = np.array([X[j] - X[i] for i in range(m) for j in range(m)])
        # ids = [i*m + j for i in range(m) for j in range(m)]

        # Change the OG array now
        X2 = X  # Faster RAM access
        X = np.array([X2[i] for i in range(m) for _ in range(m)])

    # 3. Rotate distances 90 degrees (x, y) = (-y, x)
    v = np.column_stack((-dX[:, 1], dX[:, 0]))

    # 4. Normalise each distance
    n = np.sqrt(np.power(v[:, 0], 2) + np.power(v[:, 1], 2))

    # 5. Distance = x1 dot n_hat
    # Preprocess 0s, this will be at least 50% of the data.
    print('Deleting null entries...')
    indices = np.where(n == 0)
    indices = list(indices[0])  # Type conversion before iterating speeds up
    # To stop division by 0, we want to have r = 0 and theta = 0, these values will be removed LATER, but kept for now
    n[indices] = 1
    v[indices] = 0

    n_hat = v / n[:, None]  # Works
    print(np.shape(X), np.shape(n_hat))
    # 6. Equivalent to a matrix dot product
    r = np.sum(X[0:, :] * n_hat, axis=1)
    theta = np.arctan2(v[:, 1], v[:, 0]) * (180 / math.pi)
    H = np.column_stack((r, theta))

    print("Finished H-space")
    return H


def calc_hough_transform(data_State, origin=(0, 0), plot=True):
    """
    :param data_input: Centroid Data - Multidimensional Array
    :param origin: Optional centre for transform
    :param plot: Bool
    :return: vector array of hough transformed input
    """
    x_values = data_State.x
    y_values = data_State.y
    frame_values = data_State.z
    X = data_State.X
    X = np.array(X)

    # Check if we have pre-computed the data
    tH1, tH2, tH3 = 0, 0, 0  # Initialisation
    try:
        H = unpickle_state("h_space{}{}".format(data_State.cover, data_State.tp))
        tH1 = H.x
        tH2 = H.y
        z = H.z
    except FileNotFoundError:
        # Apply Hough transforms if data not found
        tH1 = __hough_transform2(X[:, [0, 2]], origin=origin, sequence=False)  # From (x, t)
        tH2 = __hough_transform2(X[:, [1, 2]], origin=origin, sequence=False)  # From (y, t)
        # tH3 = __hough_transform2(X[:, [0, 1]], origin=origin, sequence=False)  # From (x, y)
        z = (tH1[:, 1] + tH2[:, 1]) / 2  # The midpoint angle between the two dimensions
        tH1 = tH1[:, 0]  # Only r_x
        tH2 = tH2[:, 0]  # Only r_y
        # The midpoint angle between the two dimensions
        pickle_state(tH1, tH2, z, path="h_space{}{}".format(data_State.cover, data_State.tp))

    if plot:
        """
        plt.scatter(tH1[:, 0], tH1[:, 1], s=1, marker="x", label="(x, t)")
        plt.scatter(tH2[:, 0], tH2[:, 1], s=1, marker="x", label="(y, t)")
        plt.title("Hough Transformed space for cover 00")
        plt.xlabel("Distance")
        plt.ylabel("Angle in degrees")
        plt.legend(markerscale=2, loc='center left', bbox_to_anchor=(1, 0.45))
        plt.show()
        """
        # This will plot the graph in 3D
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        n = 100000  # Selection
        z = (tH1[:n, 1] + tH2[:n, 1]) / 2
        ax.scatter(tH1[:n, 0], tH2[:n, 0], z, s=1)

        ax.set_xlabel('$r_x$ Axes')
        ax.set_ylabel('$r_y$ Axes')
        ax.set_zlabel('Theta')
        plt.title("H-space intermediate for cover 00")
        plt.show()

    return tH1, tH2, z


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


def linear_velocities(data_State, fps_input, mu_input, distance_input=None):
    # lv = dist(i, j) * fps * mu * (1/N-1) for frame N, j is the sperm in frame N I think??
    # Time complexity is really high since it requires each sperm being compared to each other sperm for every frame
    # Not yet finished.
    dist = distance_input
    if distance_input is None:
        dist = calc_distances(data_State, 2, True, frame_diff=1)
    lv = []
    # print(get_distance(data, frame_n=0, i=0, j=0, frame_diff=1))
    # Obviously not finished yet
    return 0


def correct_for_missing(smallest_indices, missing_indices):
    predictions = []
    counter = 0
    for index in smallest_indices:
        temp = index
        while (temp + counter) in missing_indices:
            counter += 1
        predictions.append(temp + counter)
    return predictions


# Returns pred_y, n_max (number of bins required)
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
    n_max = len(data_State.X2[0])
    previous_n = n_max
    for frame in data_State.X2:
        if len(frame) > previous_n:
            n_max += len(frame) - previous_n
        previous_n = len(frame)

    # Could make unique colours for each cluster, but randomising is a good compromise
    # (Generating DISTINCT n colours is a time-complex task)
    colors = [[random.uniform(0, 0.7), random.uniform(0, 0.7), random.uniform(0, 0.7)] for i in range(n_max)]

    # List comprehensions are over 100% more efficient
    clusters = [[i, []] for i in range(n_max)]

    # Append all of the clustering predictions to the data structure
    for i, frame in enumerate(pred_y):
        for j, prediction in enumerate(frame):
            clusters[prediction][1].append([data_State.X2[i][j][0],
                                            data_State.X2[i][j][1],
                                            i])

    # List comprehensions are over 100% more efficient
    clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n_max) if len(clusters[i][1]) != 0]

    n_clusters_out = len(clusters_asarr)

    # Flatten pred_y
    pred_y = [item for sublist in pred_y for item in sublist]

    return pred_y, n_clusters_out


def import_data(acceptable_tp, cover='00', tp='49', verbose=False):
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
        print('Importing video Data for tp {} cover {}'.format(tp, cover))
    with open(r"mojo_sperm_tracking_data_bristol\tp{}\cover{}_YOLO_NO_TRACKING_output\centroids_with_meta.json".format(
            tp, cover_),
            "r") as temp_read:
        data_ = json.load(temp_read)
    if verbose:
        print('Data import completed.')
    return data_


def produce_histogram(cover='00', tp='49', verbose=False, draw_type='bar', bin_count=25, outlierline=False):
    """
    :param bin_width: int (width of bins rescaling factor)
    :param bin_count: int (num of bins)
    :param draw_type: str (hisstype)
    :param cover: cover number
    :param tp: tp number
    :param verbose: bool, debug
    :return: None
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

    distances = [min(_list) for _list in distances]  # When the number decreases, we want to throw away that min index
    # distances = _flatten_lists(_flatten_lists(distances))
    # Remove the initial superficial zeroes. Theoretically, you can apply demoivre's and speed this up with list comp
    distances_temp = []
    flag = 0 if distances[0] == 0 else 1
    for counter, element in enumerate(distances):
        if not flag:
            if element != 0:
                flag = 1
        else:
            distances_temp.append(element)
    distances = distances_temp

    # distances = [element for element in distances if element != 0]
    # print(distances)
    plt.figure(figsize=(7.8, 5))  # Set the sizing

    # Recalculate the bins TWICE, to remove bins containing 1 data point, so we can ignore outliers
    # The outliers ARE outliers,
    # because they are generated when a sperm leaves frame and a new one enters within the SAME frame
    # Flawless trimming of outliers by purely observing the data is provably impossible.
    # My suggestion would be to improve the AI used to detect sperms
    # Alternative approach:
    # https://stackoverflow.com/questions/51329109/histogram-hide-empty-bins/51333497

    _, bins, _ = plt.hist(distances, bin_count, alpha=0.75, density=True)
    plt.clf()
    n, bins, patches = plt.hist(distances, bin_count, range=(0, bins[5]), alpha=0.75, histtype=draw_type_t)

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

    plt.title('Histogram for the distances travelled by sperms between frames for tp {} and cover {}'.format(tp, cover))
    plt.xlabel('Distance travelled in 1 frame')
    plt.ylabel('Frequency (Total {})'.format(total))
    plt.grid()
    plt.rc('axes', axisbelow=True)  # So the grid goes behind the lines
    plt.show()


# ~~~~~ Main ~~~~~
# We could pull the data all from a github using pygithub and PAT keys/SSH keys
# However, I am unsure if their data is copyrighted
# In which case we would need a license to upload their data to github.
# Obviously change your path to wherever you want to save the cover0_0
# len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
def run_main(tp='49', cover='00', plot=False, algorithm='dbscan', verbose=False, plot_type='2d'):

    data = import_data(acceptable_tpG, cover=cover, tp=tp, verbose=verbose)

    # export_to_excel(data, "MDM3_MOJO2")  # Export data to a spreadsheet

    # print(data["extra_information"])  # Resize factor for the data
    num_frames(data)  # Calc the number of frames for the data. Might not always be 301
    fps = 60  # The frames per second for the all the data is 60 per second
    mu = 0.13  # The pixel pitch in microns (As given by Mojo)
    data = State2(data, cover=cover, tp=tp)

    # Cluster and plot the clusters for their 2D projections
    calc_clusters(data, plot=plot, algorithm=algorithm, plot_type=plot_type)


# Globals
acceptable_tpG = ['49', '57']  # Acceptable TP covers
acceptable_draw_typesG = ['bar', 'step', 'stepfilled']  # 2021/02 These are the useful only ones we want
# run_main(algorithm=input("'kmeans', 'dbscan', 'mike', 'none', 'hdbscan', 'gmm', 'htdbscan', 'richard' ? "))
# run_main(algorithm='dbscan', cover='13', plot=True, verbose=True, plot_type='bar_graph')
# run_main(algorithm='dbscan', cover='13', plot=True, verbose=True, plot_type='3d')

produce_histogram(draw_type='bar', bin_count=100, cover='00')