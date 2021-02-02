# Import Statements
import json
import math
import numpy as np
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

# For profiling the code
import cProfile
import re


# ~~~~~ Functions Library (I would make this a seperate file, but cba coz we're using github and nobody links gits)
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


def calc_clusters(data_input, algorithm="kmeans", n_clusters=10, plot=True):
    valid_algorithms = ["kmeans", "dbscan", "mikes", "none", "hdbscan", "gmm"]
    algorithm_t = "dbscan"
    if algorithm in valid_algorithms:
        algorithm_t = algorithm

    # Find all positions of center of sperm heads for data video
    # Plots the graph by default.
    # Returns the clusters
    x_values, y_values, frame_values = [], [], []

    for i in range(len(data_input["centroids"])):  # number of frames
        for j in range(len(data_input["centroids"][i])):  # number of sperms in frame
            x_values.append(data_input["centroids"][i][j]["center"][0])
            y_values.append(data_input["centroids"][i][j]["center"][1])
            frame_values.append(i)

    X = []
    for i in range(len(x_values)):
        X.append([x_values[i], y_values[i], frame_values[i]])
    X = np.asarray(X)  # The data

    # Calculate the predictions using dbscan
    # eps: max distance to be considered a cluster neighbour.
    # min_samples: similar to KNN, minimum neighbours to make a cluster
    pred_y = []  # Initialisation
    n=0  # Intialisation, should change to the number of labels
    if algorithm_t.lower() == 'dbscan':
        model = DBSCAN(eps=13, min_samples=5, n_jobs=2)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        pred_y = model.fit_predict(X)

        # How many clusters?
        n = len(set(labels)) - (1 if -1 in labels else 0)
        print('# of clusters ', n_clusters)
        print('LABELS: ', labels)

    # Calculate the predictions using kmeans
    elif algorithm_t.lower() == 'kmeans':
        n = n_clusters  # Number of clusters
        model = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=10, random_state=0)
        pred_y = model.fit_predict(X)

    # Calculate the predictions using Mike distance tracking method
    elif algorithm_t.lower() == 'mikes':
        pred_y, n_max = track_sperms(data, distances, distances_f)
        # I couldn't come up with a way to make it reuse this code, plotting and predicting is done all in-function
        return 0

    elif algorithm_t.lower() == 'hdbscan':
        # code from: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(X)
        labels = clusterer.labels_
        pred_y = clusterer.fit_predict(X)

        # How many clusters?
        n = len(set(labels)) - (1 if -1 in labels else 0)
        print('# of clusters ', n_clusters)
        print('LABELS: ', labels)

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

    elif algorithm_t.lower() == 'none':
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        c4 = 0
        x2 = []
        y2 = []
        f2 = []

        for i, frame in enumerate(data_input["centroids"]):
            for sperm in frame:
                x2.append(sperm["center"][0])
                y2.append(sperm["center"][1])
                f2.append(i)
                c4 += 1
        ax.scatter(x2, y2, f2, s=1, color='black')
        print(c4)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Frame number')
        plt.title("Sperm Centroids every frame")
        plt.show()
        # End the function
        return 0

    # Generate the ID and a random colours for each ID
    # List comprehensions are over 100% more efficient
    clusters = [[i, []] for i in range(n)]

    # Could make unique colours for each cluster, but randomising is a good compromise
    # (Generating DISTINCT n colours is a time-complex task)
    colors = [[random.uniform(0, 0.7), random.uniform(0, 0.7), random.uniform(0, 0.7)] for i in range(n)]

    # Append all of the clustering predictions to the data structure
    for i in range(len(pred_y)):
        clusters[pred_y[i]][1].append([x_values[i], y_values[i], frame_values[i]])

    # List comprehensions are over 100% more efficient
    clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n) if len(clusters[i][1]) != 0]

    if plot:
        fig, (ax3, ax2, ax1) = plt.subplots(1, 3)
        fig.set_size_inches(17, 5)  # Set the sizing
        fig.suptitle('{} clustering applied to the sperm centroids for cover 00'.format(algorithm_t.upper()),
                     fontsize=18)
        for i, cluster in enumerate(clusters_asarr):
            color_map = colors[i]
            ax1.scatter(cluster[:, 0], cluster[:, 1], label=i, color=color_map, s=5)
            ax2.scatter(cluster[:, 0], cluster[:, 2], label=i, color=color_map, s=5)
            ax3.scatter(cluster[:, 1], cluster[:, 2], label=i, color=color_map, s=5)

        if algorithm.lower() == 'kmeans':
            # Only kmeans uses cluster centroids
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
        ax1.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.45), markerscale=6)

        ax3.title.set_text(' (A) ')
        ax2.title.set_text(' (B) ')
        ax1.title.set_text(' (C) ')
        plt.show()
        """
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
        plt.title("Sperm Centroids every frame")
        plt.show()
        """

    return clusters_asarr


# data_input, power, root=True. Calculate the distance for given cover data. frame_diff is the difference between frames
def calc_distances(data_input, power=2, root=True, frame_diff=1):
    # Calculate the distance between each and every sperm for every frame, done for distance between 'frame_diff' frames
    p = float(power)

    # Overall, the relevant data is in form: data["centroids"][frame_number][sperm_number]["desired_info_on_sperm"]
    # len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
    # [FRAME 1:[[coord 1x, coord 1y], [coord 2x, coord2y], etc], FRAME2: [[],[], ...], ..., FRAME 301: ...]

    X = [[[sperm["center"][0], sperm["center"][1]] for sperm in frame] for frame in data_input["centroids"]]

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


def get_distance(data_input, frame_n, i, j, frame_diff=1, distance_input=None, power=2, root=True):
    # Return the distance between sperm i and sperm j in frame_n
    dist = distance_input
    if distance_input is None:
        dist = calc_distances(data_input, power=power, root=root, frame_diff=frame_diff)
    elif frame_n < frame_diff:
        print('Error, frame diff for distance was larger than frame_n')
        sys.exit()

    output = dist[frame_n][i][j]
    # REMEMBER THAT SOMETIMES NEW SPERMS ENTER THE FRAME, so diff might not make sense.
    # I Need to include IDs and searches instead of indexes for a perfect algorithm.
    # Issue is that this will massively slow down the algorithm. The data needs to instead be processed
    return output


def linear_velocities(data_input, fps_input, mu_input, distance_input=None):
    # lv = dist(i, j) * fps * mu * (1/N-1) for frame N, j is the sperm in frame N I think??
    # Time complexity is really high since it requires each sperm being compared to each other sperm for every frame
    # Not yet finished.
    dist = distance_input
    if distance_input is None:
        dist = calc_distances(data_input, 2, True, frame_diff=1)
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
def track_sperms(data_input, distance_input=None, distance_f_input=None, power=2, root=True, frame_diff=1, plot=True):
    # Generate the distances if not done already
    dist = distance_input
    if distance_input is None:
        dist = calc_distances(data_input, power, root, frame_diff=frame_diff)
    dist_f = distance_f_input
    if distance_f_input is None:
        dist_f = calc_distances(data_input, power, root, frame_diff=-frame_diff)

    # Return error if frame_diff is negative
    if frame_diff < 1:
        print('Error, frame_diff is negative for sperm tracking')
        sys.exit()

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
    pred_y = [list(range(len(data_input["centroids"][i]))) for i in range(frame_diff)]
    previous_n = len(data_input["centroids"][frame_diff-1])  # Previous n for comparison
    starting_len = len(data_input["centroids"][frame_diff-1])  # The starting length
    missing_indices = []
    new_sperms = []
    for frame_i, frame in enumerate(data_input["centroids"][frame_diff:], frame_diff):
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
            for sperm_j in range(len(data_input["centroids"][frame_i])):
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
            for sperm_j in range(len(data_input["centroids"][frame_i - frame_diff])):
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
    n_max = len(data_input["centroids"][0])
    previous_n = n_max
    for frame in data_input["centroids"]:
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
            clusters[prediction][1].append([data_input["centroids"][i][j]["center"][0],
                                            data_input["centroids"][i][j]["center"][1],
                                            i])

    # List comprehensions are over 100% more efficient
    clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n_max) if len(clusters[i][1]) != 0]

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        for i, cluster in enumerate(clusters_asarr):
            color_map = colors[i]
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=color_map, s=1)

        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Frame number')
        plt.title("Sperm Centroids every frame")
        plt.show()
    return pred_y, n_max


# ~~~~~ Main ~~~~~
# We could pull the data all from a github using pygithub and PAT keys/SSH keys
# However, I am unsure if their data is copyrighted
# In which case we would need a license to upload their data to github.
# Obviously change your path to wherever you want to save the cover0_0

# len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
print('Importing video Data...')
with open(r"Q:\Michael's Stuff\Eng Maths\MDM3\PhaseB- "
          r"Mojo\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\centroids_with_meta.json",
          "r") as read_file:
    data = json.load(read_file)
print('Data import completed.')

print(data["extra_information"])  # Resize factor for the data
num_frames(data)  # Calc the number of frames for the data. Might not always be 301
fps = 60  # The frames per second for the all the data is 60 per second
mu = 0.13  # The pixel pitch in microns (As given by Mojo)

# Precalculate the euclidean distances of each sperm from each other sperm for every frame with frame diff
distances = calc_distances(data, 2, True, frame_diff=1)
distances_f = calc_distances(data, 2, True, frame_diff=-1)
# linear_velocities(data, fps, mu, distance_input=distances_arr)  # Unfinished

# Plot a line between each sperm, using primitive tracking as the shortest distance sperm in the previous frame.
# Add each sperm coordinate across all the frames to a separate list
# track_sperms(data, power=2, plot=True)

# 'dbscan' runs dbscan based clustering and same with 'kmeans' for kmeans. 'mikes' with mike's physics based approach
calc_clusters(data, algorithm=input("'kmeans', 'dbscan', 'mikes', 'none', 'hdbscan', 'gmm' ? "))
# export_to_excel(data, "MDM3_MOJO2")  # Export data to a spreadsheet
