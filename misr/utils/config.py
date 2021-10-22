from tkfilebrowser import askopendirname
import numpy as np
from os.path import join as path_join
from os.path import exists
import re


def save_water_info(water_filepath):
    # The columns are:      T       Dyn. visc   Kin. visc.  Rho     expressed in [C, mPa.s, mm²/s, g/cm³]
    water_info = np.array([[2, 1.6735, 1.6736, 0.9999],
                           [3, 1.619, 1.6191, 1],
                           [4, 1.5673, 1.5674, 1],
                           [5, 1.5182, 1.5182, 1],
                           [6, 1.4715, 1.4716, 0.9999],
                           [7, 1.4271, 1.4272, 0.9999],
                           [8, 1.3847, 1.3849, 0.9999],
                           [9, 1.3444, 1.3447, 0.9998],
                           [10, 1.3059, 1.3063, 0.9997],
                           [11, 1.2692, 1.2696, 0.9996],
                           [12, 1.234, 1.2347, 0.9995],
                           [13, 1.2005, 1.2012, 0.9994],
                           [14, 1.1683, 1.1692, 0.9992],
                           [15, 1.1375, 1.1386, 0.9991],
                           [16, 1.1081, 1.1092, 0.9989],
                           [17, 1.0798, 1.0811, 0.9988],
                           [18, 1.0526, 1.0541, 0.9986],
                           [19, 1.0266, 1.0282, 0.9984],
                           [20, 1.0016, 1.0034, 0.9982],
                           [21, 0.9775, 0.9795, 0.998],
                           [22, 0.9544, 0.9565, 0.9978],
                           [23, 0.9321, 0.9344, 0.9975],
                           [24, 0.9107, 0.9131, 0.9973],
                           [25, 0.89, 0.8926, 0.997],
                           [26, 0.8701, 0.8729, 0.9968],
                           [27, 0.8509, 0.8539, 0.9965],
                           [28, 0.8324, 0.8355, 0.9962],
                           [29, 0.8145, 0.8178, 0.9959],
                           [30, 0.7972, 0.8007, 0.9956],
                           [31, 0.7805, 0.7842, 0.9953],
                           [32, 0.7644, 0.7682, 0.995],
                           [33, 0.7488, 0.7528, 0.9947],
                           [34, 0.7337, 0.7379, 0.9944],
                           [35, 0.7191, 0.7234, 0.994],
                           [36, 0.705, 0.7095, 0.9937],
                           [37, 0.6913, 0.6959, 0.9933],
                           [38, 0.678, 0.6828, 0.993],
                           [39, 0.6652, 0.6702, 0.9926],
                           [40, 0.6527, 0.6579, 0.9922],
                           [45, 0.5958, 0.6017, 0.9902],
                           [50, 0.5465, 0.5531, 0.988],
                           [55, 0.5036, 0.5109, 0.9857],
                           [60, 0.466, 0.474, 0.9832],
                           [65, 0.4329, 0.4415, 0.9806],
                           [70, 0.4035, 0.4127, 0.9778],
                           [75, 0.3774, 0.3872, 0.9748],
                           [80, 0.354, 0.3643, 0.9718]])
    # Refference: IAPWS 2008
    np.savetxt(water_filepath, water_info)


def setup():
    print("This script is used when first setting up the working script folder.")
    print("It sets up the .config file with all the configuration. The .config file MUST")
    print("be located in the same directory as the working script from which you are calling the misr module.\n")

    try:
        conf_file = open(".config", "x")
    except FileExistsError:
        ans = str(input("The config file allready exists! Do you want to overwrite it? [Y/n] "))
        if ans != "Y":
            exit()

    meas_dir = askopendirname(title="Select the Measurements folder")
    info_dir = askopendirname(title="Select the Info folder (Rod, Tub, ...)")

    print("This file contains the inforamtion needed to run scripts easier and faster.")
    print("MEAS_DIR='{0}'".format(meas_dir), file=conf_file)
    print("INFO_DIR='{0}'".format(info_dir), file=conf_file)

    water_filepath = path_join(info_dir, "water_info.dat")
    print("WATER_FILE='{0}'".format((water_filepath)), file=conf_file)

    conf_file.close()

    ans = ""
    ans = str(input("Save a new water profile? [Y/n] "))
    if ans == "Y":
        save_water_info(water_filepath)

    print("\n" + "-"*10 + " Setup all done! " + "-"*10)


def get_config():
    if exists(".config"):
        conf_file = open(".config", "r")
    else:
        print("No config file found... Making a new one!\n")
        setup()

        conf_file = open(".config", "r")

    conf_file_contents = "\n".join(conf_file.readlines())
    conf_file.close()

    meas_dir = re.search("MEAS_DIR='.*'", conf_file_contents)[0][10:-1]
    info_dir = re.search("INFO_DIR='.*'", conf_file_contents)[0][10:-1]
    water_info = re.search("WATER_FILE='.*'", conf_file_contents)[0][12:-1]

    return meas_dir, info_dir, water_info

