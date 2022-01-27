from ..utils.get_rod_and_tub_info import guess_and_get_rod_and_tub_info,\
    get_rod_info, get_tub_info, get_tub_info_from_id, get_rod_info_from_id
from gvar import gvar


def get_Rod_and_Tub(system_responses):
    # FOR NOW ONLY ONE ROD and TUB CAN BE SELECTED!

    # Can get them directly from measurement object :)
    rod_id_from_config_files = [sr.meas.rod_id for sr in system_responses]
    if len(set(rod_id_from_config_files)) == 1:
        if rod_id_from_config_files[0] is not None:
            rod_info = get_rod_info_from_id(rod_id_from_config_files[0])
        else:
            print("No rod info was found, please select one to continue!")
            rod_info = get_rod_info()
    elif len(set(rod_id_from_config_files)) == 2 and None in rod_id_from_config_files:
        print("Some measurements have data about the rod id, some dont.")
        print(f"Here is what I found {set(rod_id_from_config_files)}.")
        print("Now you choose what to do.")
        rod_info = get_rod_info()
    elif len(set(rod_id_from_config_files)) > 1:
        print("Detected using multiple rods! NOT SUPPORTED FOR NOW! EXITING!")
        exit()
    else:
        print("No rod info was found, please select one to continue!")
        rod_info = get_rod_info()

    tub_id_from_config_files = [sr.meas.tub_id for sr in system_responses]
    if len(set(tub_id_from_config_files)) == 1:
        if tub_id_from_config_files[0] is not None:
            tub_info = get_tub_info_from_id(tub_id_from_config_files[0])
        else:
            print("No tub info was found, please select one to continue!")
            tub_info = get_tub_info()
    elif len(set(tub_id_from_config_files)) == 2 and None in tub_id_from_config_files:
        print("Some measurements have data about the tub id, some dont.")
        print(f"Here is what I found {set(tub_id_from_config_files)}.")
        print("Now you choose what to do.")
        tub_info = get_tub_info()
    elif len(set(tub_id_from_config_files)) > 1:
        print("Detected using multiple tubs! NOT SUPPORTED FOR NOW! EXITING!")
        exit()
    else:
        print("No tub info was found, please select one to continue!")
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
