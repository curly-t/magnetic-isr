from ..utils.get_rod_and_tub_info import guess_and_get_rod_and_tub_info, get_rod_info, get_tub_info


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
        self.L = rod_info[1]
        self.m = rod_info[2]
        self.d = rod_info[3]
        self.L_err = rod_info[4]
        self.m_err = rod_info[5]
        self.d_err = rod_info[6]


class Tub:
    def __init__(self, tub_info):
        # The fields are: (every value written in each row, all in one column)
        # id, W, h, V, W_err, h_err, V_err     (id, Width, hight, Volume (NOT total but water volume used  [SI units]))
        self.id = tub_info[0]
        self.W = tub_info[1]
        self.h = tub_info[2]
        self.V = tub_info[3]
        self.W_err = tub_info[4]
        self.h_err = tub_info[5]
        self.V_err = tub_info[6]
