# Import Statements
import tracking_algorithm48 as TP
import math
import sys
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# U Value- Analysis. U is a measure of how far a clustering is from equal distribution
# U is defined as: U = Mean(|n_f - x|) where x is the number of points in each cluster
algorithm_comparisons_key = ['none', 'kmeans', 'dbscan', 'hdbscan', 'gmm', 'mike', 'mike-htdbscan', 'richard-dbscan']


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


def plot_all_04(_algorithm_comparisons):
    _tp = '49'
    _cover = '04'
    for algorithm in algorithm_comparisons_key:
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


if __name__ == '__main__':
    # Compute the different evaluation measures for different algorithms. Here are examples:
    # eval_U(algorithm_comparisons_key)
    # eval_convex_hull(algorithm_comparisons_key)
    # eval_runtime(algorithm_comparisons_key)
    # eval_convex_hull(['ground-truth'])
    # eval_U(['ground-truth'])
    # plot_all_04(algorithm_comparisons_key)
    pass
