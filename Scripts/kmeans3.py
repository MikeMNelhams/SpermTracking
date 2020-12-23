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

# We could pull the data all from a github using pygithub and PAT keys/SSH keys
# However, I am unsure if their data is copyrighted
# In which case we would need a license to upload their data to github.

# Obviously change your path to wherever you want to save the cover0_0
with open(r"C:\Users\Mike Nelhams\Documents\Mike's Stuff 2\MDM3-B\mojo_sperm_tracking_data_bristol\mojo_sperm_tracking_data_bristol\tp49\cover0_0_YOLO_NO_TRACKING_output\centroids_with_meta.json", "r") as read_file:
    data = json.load(read_file)

x_values = []
y_values = []
frame_values = []

# Plotting all positions of center of sperm heads for video 0
for i in range(0, len(data["centroids"])):  # number of frames
    for j in range(0, len(data["centroids"][i])):  # number of sperms in frame
        x_values.append(data["centroids"][i][j]["center"][0])
        y_values.append(data["centroids"][i][j]["center"][1])
        frame_values.append(i)

# Excel file
with xlsxwriter.Workbook('MDM3_MOJO2.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for i in range(len(x_values)):
        worksheet.write(i, 0, x_values[i])
        worksheet.write(i, 1, y_values[i])
        worksheet.write(i, 2, frame_values[i])

# Richard's Code (Edited by Mike)
X = []
for i in range(0,len(x_values)):
    X.append([x_values[i], y_values[i], frame_values[i]])
X = np.asarray(X)  # The data

n = 12  # Number of clusters
# Generate the ID and a random colours for each ID
# List comprehensions are over 100% more efficient
clusters = [[i, []] for i in range(n)]

# Could make unique colours for each cluster, but randomising is a good compromise
# (Generating DISTINCT n colours is a time-complex task)
colors = [[random.uniform(0,0.7), random.uniform(0,0.7), random.uniform(0,0.7)] for i in range(n)]

# Calculate the predictions using kmeans
model = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=10, random_state=0)
pred_y = model.fit_predict(X)

# Calculate the predictions using dbscan
# eps: max distance to be considered a cluster neighbour.
# min_samples: similar to KNN, minimum neighbours to make a cluster
"""
model = DBSCAN(eps=100, min_samples=2)
pred_y = model.fit_predict(X)
"""

# Append all of the clustering predictions to the data structure
for i in range(0, len(pred_y)):
    clusters[pred_y[i]][1].append([x_values[i], y_values[i], frame_values[i]])

# List comprehensions are over 100% more efficient
clusters_asarr = [np.asarray(clusters[i][1]) for i in range(n)]

fig = plt.figure()
ax = plt.axes(projection="3d")
for i, cluster in enumerate(clusters_asarr):
    color_map = colors[i]
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=color_map, s=1)

ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], s=100, color=colors)
ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Frame number')
plt.title("Sperm Centroids every frame")
plt.show()
