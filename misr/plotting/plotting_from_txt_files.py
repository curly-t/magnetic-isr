import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def plot_single_response(resp, label, axes, plot_params):
    axes[0].errorbar(np.log10(resp[:, 0]), np.log10(resp[:, 1]),
                     xerr=resp[:, 3] / (np.log(10) * resp[:, 0]), yerr=resp[:, 4] / (np.log(10) * resp[:, 1]),
                     label=label, **plot_params)
    axes[1].errorbar(np.log10(resp[:, 0]), resp[:, 2],
                     xerr=resp[:, 3] / (np.log(10) * resp[:, 0]), yerr=resp[:, 5],
                     label=label, **plot_params)


def plot_responses(responses):
    """ Plots the responses gotten from the txt files - there is no data objects, just pure data.
        responses ---- a dictionary of 2D numpy arrays (shown below), of all data selected in form of .txt files.
        {"label": np.array([[freq AR phase]1, [freq AR phase]2, ...])}"""

    plot_params = {"elinewidth": 1, "capthick": 1, "capsize": 2, "markersize": 6, "linewidth": 0.5}

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    prop_cycler = (cycler(marker=['o', 'v', '^', '<', '>', 's', 'p', 'D']) *
                   cycler(color=['r', 'g', 'b', 'y', 'k', 'c', 'm']))

    for (label, response), plot_props in zip(responses.items(), prop_cycler):
        plot_params.update(plot_props)
        plot_single_response(response, label[:-4], axes, plot_params)

    axes[0].set_ylabel(r'$\log_{10}(AR)$   (AR [m/A])', fontsize=14)
    axes[0].legend(fontsize=7)
    axes[0].grid()

    axes[1].set_xlabel(r'$\log_{10}(\nu)$', fontsize=14)
    axes[1].set_ylabel(r'$Phase$', fontsize=14)
    axes[1].grid()

    plt.tight_layout()
    plt.show()