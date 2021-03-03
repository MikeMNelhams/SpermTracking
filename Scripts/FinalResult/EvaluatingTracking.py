# Import Statements
import tracking_algorithm54 as TP
import math
import sys
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Scipy
import scipy as sp
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# Stonesoup
from stonesoup.types.array import Matrix
from stonesoup.types.array import StateVector
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.metricgenerator import ospametric

# U Value- Analysis. U is a measure of how far a clustering is from equal distribution
# U is defined as: U = Mean(|n_f - x|) where x is the number of points in each cluster


def convex_hull_density(data_X):
    volume = convex_hull_volume(data_X)
    density = volume / data_X.size
    return density


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6


def convex_hull_volume(pts):
    ch = ConvexHull(pts)
    dt = Delaunay(pts[ch.vertices])
    tets = dt.points[dt.simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))


def plot_all(_algorithm_comparisons, cover='04', tp='49'):
    _tp = tp
    _cover = cover
    for algorithm in _algorithm_comparisons:
        TP.run_main(algorithm=algorithm, tp=_tp, cover=_cover, plot=True, write_output=False, verbose=False)


def eval_U(_algorithm_comparisons):
    means = []
    stds = []
    num_algorithms = len(_algorithm_comparisons)
    for count, algorithm in enumerate(_algorithm_comparisons):
        print('Algorithm: {} of {}'.format(count + 1, num_algorithms))
        if algorithm == 'ground-truth':
            mean_, std_ = TP.evaluate_U_success(algorithm='ground-truth', tps=['49'], plot=False, mean_line=True,
                                                verbose=False, bar_width=0.4, bar_gap_scale=1, max_u_value=35)
        else:
            mean_, std_ = TP.evaluate_U_success(algorithm=algorithm, tps=['49', '57'], plot=False, mean_line=True,
                                                verbose=False, bar_width=0.4, bar_gap_scale=1, max_u_value=35)
        means.append(mean_)
        stds.append(std_)

    print(means, stds)


def eval_convex_hull(_algorithm_comparisons):
    num_algorithms = len(_algorithm_comparisons)
    tps = ['49', '57']
    mean_densities = []
    for count, algorithm in enumerate(_algorithm_comparisons):
        print('Algorithm: {} of {}'.format(count + 1, num_algorithms))
        densities = []
        if algorithm == 'ground-truth':
            # Dummy State object. Data only exists for cover '04' tp '49'
            data = TP.import_data(TP.acceptable_tpG, cover='04', tp='49', verbose=False)
            data = TP.State2(data, cover='04', tp='49')
            clusters, _, _ = TP.calc_clusters(data, algorithm='ground-truth', plot=False, write_output=False, verbose=False,
                                              return_clusters=True)
            for cluster in clusters:
                # Minimum of 4 points to define a 3d shape. Since 3 points is always a plane
                # Funnily enough it's more computationally expensive to check if they lie on the same plane
                #   than to just calculate the convex hull density.
                #   (Since checking N points for coplanarity requires checking every combination of 4 points.)
                # So let's use Try Except to cheat it a bit
                if len(cluster[1]) > 3:
                    try:
                        density = convex_hull_density(np.array(cluster[1]))
                        densities.append(density)
                    except sp.spatial.qhull.QhullError:
                        pass

        else:
            for tp in tps:
                for cover in TP.acceptable_coversG:
                    data = TP.import_data(TP.acceptable_tpG, cover=cover, tp=tp, verbose=False)
                    data = TP.State2(data, cover=cover, tp=tp)
                    clusters, _, _ = TP.calc_clusters(data, algorithm=algorithm, plot=False, write_output=False, verbose=False,
                                                      return_clusters=True)
                    for cluster in clusters:
                        # Minimum of 4 points to define a 3d shape. Since 3 points is always a plane
                        # Funnily enough it's more computationally expensive to check if they lie on the same plane
                        #   than to just calculate the convex hull density.
                        #   (Since checking N points for coplanarity requires checking every combination of 4 points.)
                        # So let's use Try Except to cheat it a bit
                        if len(cluster[1]) > 3:
                            try:
                                density = convex_hull_density(np.array(cluster[1]))
                                densities.append(density)
                            except sp.spatial.qhull.QhullError:
                                pass

        mean_density = np.mean(np.array(densities))
        mean_densities.append(mean_density)
    print(mean_densities)
    return mean_densities


def eval_runtime(_algorithm_comparisons):
    num_algorithms = len(_algorithm_comparisons)
    tps = ['49', '57']
    mean_runtimes = []
    for count, algorithm in enumerate(_algorithm_comparisons):
        print('Algorithm: {} of {}'.format(count + 1, num_algorithms))
        runtimes = []
        for tp in tps:
            for cover in TP.acceptable_coversG:
                data = TP.import_data(TP.acceptable_tpG, cover=cover, tp=tp, verbose=False)
                data = TP.State2(data, cover=cover, tp=tp)
                start = datetime.now()
                TP.calc_clusters(data, algorithm=algorithm, plot=False, write_output=False, verbose=False,
                                 return_clusters=False)
                time_delta = datetime.now() - start
                runtimes.append(time_delta.microseconds)
        mean_runtimes.append(np.mean(np.array(runtimes)))
    print('Mean Runtimes: ', mean_runtimes)


def eval_OPSA(_algorithm_comparisons, cutoff=100, p=2):
    # Test it with 'kmeans' since this is the easier one
    if cutoff <= 0:
        print('cutoff distance must be a positive float, not {}'.format(cutoff))
        sys.exit(1)

    algorithm = 'kmeans'
    cover = '04'
    tp = '49'

    # Predict the algorithm clusters
    data = TP.import_data(TP.acceptable_tpG, cover=cover, tp=tp, verbose=False)
    data = TP.State2(data, cover=cover, tp=tp)
    predicted_clusters, _, _ = TP.calc_clusters(data, algorithm=algorithm, plot=False, write_output=False, verbose=False,
                                                             return_clusters=True)
    # Collect the actual clusters
    # Dummy State object. Data only exists for cover '04' tp '49'
    data = TP.import_data(TP.acceptable_tpG, cover='04', tp='49', verbose=False)
    data = TP.State2(data, cover='04', tp='49')
    ground_truth_clusters, _, _ = TP.calc_clusters(data, algorithm='ground-truth', plot=False, write_output=False, verbose=False,
                                                            return_clusters=True)

    # Convert all info into two np.arrays [x, y, frame, ID] sorted by frame
    # [cluster[1][0], cluster[1][1], cluster[1][2], cluster[0]]
    # predicted_data = [cluster for cluster in predicted_clusters]
    # print(predicted_data)
    predicted_data = [[sperm[0], sperm[1], sperm[2], cluster[0]] for cluster in predicted_clusters for sperm in cluster[1]]
    predicted_data = np.array(predicted_data)

    ground_truth_data = [[sperm[0], sperm[1], sperm[2], cluster[0]] for cluster in ground_truth_clusters for sperm in cluster[1]]
    ground_truth_data = np.array(ground_truth_data)

    # Data cleanup
    del predicted_clusters
    del ground_truth_clusters

    m = predicted_data.shape[0]
    n = ground_truth_data.shape[0]
    if not m and not n:
        # If the cardinalities are 0, then the distance is 0
        return 0

    base_distance = __base_distance(predicted_data, ground_truth_data, p, 1)
    c_distance = min(cutoff, base_distance)
    OSPA_distance = (((1 / n) * c_distance) ** p + ((n - m) * (cutoff ** p))) ** (1/p)
    print(OSPA_distance)


def __base_distance(predicted_data, ground_truth_data, p, alpha):
    if not (0 <= alpha <= 1):
        print('Alpha must be a real float between 0 and 1 both inclusive')
        sys.exit(1)
    d1 = np.power(__distance(predicted_data[:, [0, 1]] - ground_truth_data[:, [0, 1]], p), p)
    d2 = np.power(__kronecker_sum(predicted_data[:, [3]], ground_truth_data[:, [3]], alpha), p)
    d = np.power(d1 + d2, 1/p)
    return d


def __distance(points, p):
    return np.linalg.norm(points, p)


def __kronecker_sum(l, s, alpha):
    """
    1 if the same labels, 0 if they are different. Sum the total kronecker complements
    len(l) == len(s) should be True
    :param l: labels1 list
    :param s: labels2 list
    :param alpha: float (0 <= alpha <= 1) ((penalty))
    :return: float
    """
    k_sum = 0
    for label1, label2 in zip(l, s):
        if label1 != label2:
            k_sum += 1
    return alpha * k_sum


algorithm_comparisons_key = ['none', 'kmeans', 'dbscan', 'hdbscan', 'gmm', 'mike', 'mike-htdbscan',
                             'richard-dbscan']

if __name__ == '__main__':
    # Compute the different evaluation measures for different algorithms. Here are examples:
    # eval_U(algorithm_comparisons_key)
    # eval_convex_hull(algorithm_comparisons_key)
    # eval_runtime(algorithm_comparisons_key)
    eval_runtime(['richard-bic-hdbscan'])
    # eval_U(['richard-bic-hdbscan'])
    # eval_OPSA(algorithm_comparisons_key)
    # plot_all(algorithm_comparisons_key, cover='04', tp='49')
    # plot_all('IDL', cover='04', tp='49')
    # plot_all(['ground-truth'], cover='04', tp='49')
    pass
