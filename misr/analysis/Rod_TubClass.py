from ..utils.get_rod_and_tub_info import guess_and_get_rod_and_tub_info, get_rod_info, get_tub_info
from gvar import gvar


def get_Rod_and_Tub(dirnames=None):
    if dirnames is not None:
        rod_info, tub_info = guess_and_get_rod_and_tub_info(dirnames)
    else:
        rod_info = get_rod_info()
        tub_info = get_tub_info()

    return Rod(rod_info), Tub(tub_info)


class Rod:
    def __init__(self, rod_info):
        # The fields are: (every value written in each row, all in one column)
        # id, L, m, d, L_err, m_err, d_err     (id, Length, mass, diameter) (ALL IN SI units!)
        self.id = rod_info[0]
        self.L = gvar(rod_info[1], rod_info[4])
        self.m = gvar(rod_info[2], rod_info[5])
        self.d = gvar(rod_info[3], rod_info[6])


class Tub:
    def __init__(self, tub_info):
        # The fields are: (every value written in each row, all in one column)
        # id, W, h, V, W_err, h_err, V_err     (id, Width, hight, Volume (NOT total but water volume used  [SI units]))
        self.id = tub_info[0]
        self.W = gvar(tub_info[1], tub_info[4])
        self.h = gvar(tub_info[2], tub_info[5])
        self.V = gvar(tub_info[3], tub_info[6])
