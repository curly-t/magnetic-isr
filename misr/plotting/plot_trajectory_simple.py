from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfiles

from ..analysis.import_trackdata import import_filepaths
from ..utils.config import get_config


def plot_traj():
    initdir = get_config()["meas"]

    filepaths = list(askopenfiles(initialdir=initdir, filetypes=["Data {.dat}"]))
    measurements = import_filepaths(filepaths)

    for meas in measurements:
        plt.plot(meas.times, meas.positions)
    plt.show()


def plot_bright():
    initdir = get_config()["meas"]

    filepaths = list(askopenfiles(initialdir=initdir, filetypes=["Data {.dat}"]))
    measurements = import_filepaths(filepaths)

    for meas in measurements:
        plt.plot(meas.times, meas.brights)
    plt.show()

