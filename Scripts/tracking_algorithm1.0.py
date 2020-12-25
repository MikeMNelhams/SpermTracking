# Import Statements
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import sys  # For halting runtime
from mpl_toolkits.mplot3d import axes3d

# Machine Learning Imports
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# For working with excel
import xlsxwriter


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
    valid_algorithms = ["kmeans", "dbscan"]
    algorithm_t = "dbscan"
    if algorithm in valid_algorithms:
        algorithm_t = algorithm

    # Find all positions of center of sperm heads for data video
    # Plots the graph by default.
    # Returns the clusters
    x_values, y_values, frame_values = [], [], []

    for i in range(0, len(data_input["centroids"])):  # number of frames
        for j in range(0, len(data_input["centroids"][i])):  # number of sperms in frame
            x_values.append(data_input["centroids"][i][j]["center"][0])
            y_values.append(data_input["centroids"][i][j]["center"][1])
            frame_values.append(i)

    X = []
    for i in range(0, len(x_values)):
        X.append([x_values[i], y_values[i], frame_values[i]])
    X = np.asarray(X)  # The data

    # Calculate the predictions using dbscan
    # eps: max distance to be considered a cluster neighbour.
    # min_samples: similar to KNN, minimum neighbours to make a cluster
    if algorithm_t.lower() == 'dbscan':
        model = DBSCAN(eps=13, min_samples=5)
        model_trained = model.fit(X)
        labels = model_trained.labels_
        pred_y = model.fit_predict(X)

        # How many clusters?
        sample_cores = np.zeros_like(labels, dtype=bool)
        sample_cores[model.core_sample_indices_] = True
        n = len(set(labels)) - (1 if -1 in labels else 0)
        print('# of clusters ', n_clusters)
        print('LABELS: ', labels)

    # Calculate the predictions using kmeans
    elif algorithm_t.lower() == 'kmeans':
        n = n_clusters  # Number of clusters
        model = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=10, random_state=0)
        pred_y = model.fit_predict(X)

    # Generate the ID and a random colours for each ID
    # List comprehensions are over 100% more efficient
    clusters = [[i, []] for i in range(n)]

    # Could make unique colours for each cluster, but randomising is a good compromise
    # (Generating DISTINCT n colours is a time-complex task)
    colors = [[random.uniform(0, 0.7), random.uniform(0, 0.7), random.uniform(0, 0.7)] for i in range(n)]

    # Append all of the clustering predictions to the data structure
    for i in range(0, len(pred_y)):
        clusters[pred_y[i]][1].append([x_values[i], y_values[i], frame_values[i]])

    # List comprehensions are over 100% more efficient
    clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n)]

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        for i, cluster in enumerate(clusters_asarr):
            color_map = colors[i]
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=color_map, s=1)

        if algorithm.lower() == 'kmeans':
            # DBSCAN does not use cluster centres
            ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], s=100,
                       color=colors)

        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Frame number')
        plt.title("Sperm Centroids every frame")
        plt.show()

    return clusters_asarr


# data_input, power, root=True. Calculate the distance for given cover data. frame_diff is the difference between frames
def calc_distances(data_input, power=2, root=True, frame_diff=1):
    # Calculate the distance between each and every sperm for every frame, done for distance between 'frame_diff' frames
    p = float(power)

    # Overall, the relevant data is in form: data["centroids"][frame_number][sperm_number]["desired_info_on_sperm"]
    # len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
    # [FRAME 1:[[coord 1x, coord 1y], [coord 2x, coord2y], etc], FRAME2: [[],[], ...], ..., FRAME 301: ...]

    X = [[[sperm["center"][0], sperm["center"][1]] for sperm in frame] for frame in data_input["centroids"]]
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
                for frame_i in range(frame_diff, len(X))]
    else:
        dist = [[[(sperm[0] - sperm2[0]) ** p + (sperm[1] - sperm2[1]) ** p
                  for sperm2 in X[frame_i - frame_diff]]
                 for sperm in X[frame_i]]
                for frame_i in range(frame_diff, len(X))]
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


# ~~~~~ Main ~~~~~
# We could pull the data all from a github using pygithub and PAT keys/SSH keys
# However, I am unsure if their data is copyrighted
# In which case we would need a license to upload their data to github.
# Obviously change your path to wherever you want to save the cover0_0

# len(data["centroids"][i]) is the number of sperms in a frame. This is subject to change each frame
print('Importing video Data...')
with open(r"C:\Users\Mike Nelhams\Documents\Mike's Stuff "
          r"2\MDM3-B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49"
          r"\cover0_0_YOLO_NO_TRACKING_output\centroids_with_meta.json", "r") as read_file:
    data = json.load(read_file)
print('Data import completed.')

# 'dbscan' runs dbscan based clustering and same with 'kmeans' for kmeans
# calc_clusters(data, algorithm=input("'kmeans' or 'dbscan'? "))
# export_to_excel(data, "MDM3_MOJO2")  # Export data to a spreadsheet
print(data["extra_information"])
num_frames(data)  # Calc the number of frames for the data. Might not always be 301
fps = 60  # The frames per second for the all the data is 60 per second
mu = 0.13  # The pixel pitch in microns (As given by Mojo)

# Precalculate the euclidean distances of each sperm from each other sperm for every frame with frame diff
distances = calc_distances(data, 2, True, frame_diff=1)
# linear_velocities(data, fps, mu, distance_input=distances_arr)  # Unfinished

# Plot a line between each sperm, using primitive tracking as the shortest distance sperm in the previous frame.
# Add each sperm coordinate across all the frames to a separate list
sperm_tracks = []

