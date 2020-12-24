# Import Statements
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Machine Learning Imports
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# For working with excel
import xlsxwriter

# ~~~~~ Functions Library (I would make this a seperate file, but cba coz we're using github and nobody links gits)
def export_to_excel(data, filename):
    # Export the data to an xlsx spreadsheet.
    with xlsxwriter.Workbook(filename + ".xlsx") as workbook:
        worksheet = workbook.add_worksheet()

    x_values, y_values, frame_values = [], [], []

    for i in range(0, len(data["centroids"])):  # number of frames
        for j in range(0, len(data["centroids"][i])):  # number of sperms in frame
            x_values.append(data["centroids"][i][j]["center"][0])
            y_values.append(data["centroids"][i][j]["center"][1])
            frame_values.append(i)

    for i in range(len(x_values)):
        worksheet.write(i, 0, x_values[i])
        worksheet.write(i, 1, y_values[i])
        worksheet.write(i, 2, frame_values[i])

    return 0


def calc_clusters(data, algorithm="kmeans",n_clusters=10,plot=True):
    # Find all positions of center of sperm heads for data video
    # Plots the graph by default.
    # Returns the clusters
    x_values, y_values, frame_values = [], [], []

    for i in range(0, len(data["centroids"])):  # number of frames
        for j in range(0, len(data["centroids"][i])):  # number of sperms in frame
            x_values.append(data["centroids"][i][j]["center"][0])
            y_values.append(data["centroids"][i][j]["center"][1])
            frame_values.append(i)

    X = []
    for i in range(0, len(x_values)):
        X.append([x_values[i], y_values[i], frame_values[i]])
    X = np.asarray(X)  # The data

    # Calculate the predictions using dbscan
    # eps: max distance to be considered a cluster neighbour.
    # min_samples: similar to KNN, minimum neighbours to make a cluster
    if algorithm.lower() == 'dbscan':
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
    elif algorithm.lower() == 'kmeans':
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

# ~~~~~ Main ~~~~~
# We could pull the data all from a github using pygithub and PAT keys/SSH keys
# However, I am unsure if their data is copyrighted
# In which case we would need a license to upload their data to github.
# Obviously change your path to wherever you want to save the cover0_0
with open(r"C:\Users\Mike Nelhams\Documents\Mike's Stuff 2\MDM3-B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\centroids_with_meta.json", "r") as read_file:
    data = json.load(read_file)

# 'dbscan' runs dbscan based clustering and same with 'kmeans' for kmeans
calc_clusters(data, algorithm=input("'kmeans' or 'dbscan'? "))
export_to_excel(data, "MDM3_MOJO2")  # Export data to a spreadsheet
