import tracking_algorithm40 as tr
import matplotlib.pyplot as plt
import numpy as np


def _r_and_t(acceptable_tpG, cover_number, tp='49', verbose=False):
    r = tr.produce_histogram(bin_count=100, cover=cover_number, cutoff=-2, plot=False, tp=tp,
                             verbose=verbose)
    t = tr.num_points(acceptable_tpG, cover=cover_number, tp=tp)
    if r == -1:
        r = 0
    return r, t


def calc_rms(tp='49', plot=True, verbose=False, loading_bars=True, mean_line=False):
    """
    :param tp: str (int 2 digit tp number)
    :param plot: bool
    :param verbose: bool
    :param loading_bars: bool
    :param mean_line: bool
    :return: float (mean RMS value for tp)
    """
    n = len(covers)
    r_all = []
    total_all = []
    if loading_bars:
        if verbose:
            print('Calculating R for {} covers'.format(n))
            for i, cover_number in enumerate(covers):
                r, t = _r_and_t(acceptable_tpG, cover_number, tp, verbose=verbose)
                r_all.append(r)
                total_all.append(t)
                print('Done Cover {} of {}'.format(i+1, n))
            print('All R: ', r_all)
            print('All T: ', total_all)
        else:
            print('Calculating R for {} covers'.format(n))
            for i, cover_number in enumerate(covers):
                r, t = _r_and_t(acceptable_tpG, cover_number, tp, verbose=verbose)
                r_all.append(r)
                total_all.append(t)
                print('Done Cover {} of {}'.format(i+1, n))
    else:
        if verbose:
            for i, cover_number in enumerate(covers):
                r, t = _r_and_t(acceptable_tpG, cover_number, tp, verbose=verbose)
                r_all.append(r)
                total_all.append(t)
            print('All R: ', r_all)
            print('All T: ', total_all)
        else:
            for i, cover_number in enumerate(covers):
                r, t = _r_and_t(acceptable_tpG, cover_number, tp, verbose=verbose)
                r_all.append(r)
                total_all.append(t)

    mean = np.mean(r_all)

    if plot:
        # Technically, this is deprecated, but try to stop me >:`)
        colors = [plt.cm.viridis(r_all[i] / max(r_all)) for i in range(n)]

        fig, ax = plt.subplots()
        fig.suptitle('RMS distance travelled in 1 frame for tp {}'.format(tp),
                     fontsize=18)
        ax.grid()  # Grids are very necessary for bar graphs
        ax.set_axisbelow(True)  # So the grid goes behind the lines
        label1 = 'Tp {}'.format(tp)
        for x_val, y_val, color_val in zip(covers, r_all, colors):
            ax.bar(x_val, y_val, label=label1, color=color_val)

        if mean_line:
            xlims = plt.gca().get_xlim()  # X max
            plt.hlines(mean, xlims[0], xlims[1], colors='red', linestyles='dashed', label='Mean', linewidth=2)

        ax.legend(['Mean'])
        ax.set_xlabel('Cover Number')
        ax.set_ylabel('Root Mean Distance travelled by sperms in 1 frame')
        plt.show()

    return mean


# Main
# calc_rms('49', verbose=True)
acceptable_tpG = ['49', '57']
covers = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
          '11', '12', '13', '14', '15', '16', '17', '18', '19']

calc_rms('49', verbose=False, loading_bars=True, mean_line=True)
