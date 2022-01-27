import numpy as np
from tkinter.filedialog import askopenfilename
import re
from os.path import join as path_join

from .config import get_config


def get_rod_info():
    initialdir = get_config()["info"]
    filepath = askopenfilename(initialdir=initialdir, title="Select Rod info file")
    # The fields are: (every value written in each row, all in one collumn)
    # id, L, m, d, L_err, m_err, d_err         (id, Length, mass, diameter)
    # Additional comments are saved in a comment
    return np.loadtxt(filepath)


def get_tub_info():
    initialdir = get_config()["info"]
    filepath = askopenfilename(initialdir=initialdir, title="Select Tub info file")
    # The fields are: (every value written in each row, all in one collumn)
    # id, W, h, V, W_err, h_err, V_err         (id, Width, hight, Volume (NOT total but water volume used  [SI units]))
    # Additional comments are saved in a comment
    return np.loadtxt(filepath)


def get_rod_info_from_id(rod_id):
    return np.loadtxt(path_join(get_config()["info"], "rod_{0}".format(rod_id)))


def get_tub_info_from_id(tub_id):
    return np.loadtxt(path_join(get_config()["info"], "tub_{0}".format(tub_id)))


def guess_rod_and_tub(dirnames):
    rod_ids = []
    tub_ids = []

    for dir in dirnames:
        rod_id_search = re.search("rod_\d*", dir)
        if rod_id_search is not None:
            rod_ids.append(int(rod_id_search[0][4:]))

        tub_id_search = re.search("tub_\d*", dir)
        if tub_id_search is not None:
            tub_ids.append(int(tub_id_search[0][4:]))

    rod_id = None
    tub_id = None

    if len(rod_ids) == len(dirnames) and len(set(rod_ids)) == 1:
            # Vse mape vsebujejo ime z istim rod_id
            rod_id = rod_ids[0]

    if len(tub_ids) == len(dirnames) and len(set(tub_ids)) == 1:
        # Vse mape vsebujejo ime z istim tub_id
        tub_id = tub_ids[0]

    return rod_id, tub_id


def guess_and_get_rod_and_tub_info(dirnames):
    rod_id, tub_id = guess_rod_and_tub(dirnames)

    if rod_id is not None:
        rod_info = get_rod_info_from_id(rod_id)
    else:
        rod_info = get_rod_info()
    
    if tub_id is not None:
        tub_info = get_tub_info_from_id(tub_id)
    else:
        tub_info = get_tub_info()

    return rod_info, tub_info
