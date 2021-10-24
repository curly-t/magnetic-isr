from tkinter.filedialog import asksaveasfilename, askopenfilename
from ..utils.config import get_config
from ..analysis.CalibrationClass import SimpleCalibration, FDMCalibration
import pickle


def save_simple_calibration(calibration):
    initialdir = get_config()["cal"]
    filepath = asksaveasfilename(initialdir=initialdir, defaultextension=".simpcal")

    with open(filepath, "wb") as cal_file:
        pickle.dump(calibration, cal_file)

    
def get_simple_calibration():
    initialdir = get_config()["cal"]
    filepath = askopenfilename(initialdir=initialdir, filetypes=["SimpleCalibration {.simpcal}"])

    with open(filepath, "rb") as cal_file:
        calibration = pickle.load(cal_file)

    return calibration


def save_FDM_calibration(calibration):
    initialdir = get_config()["cal"]
    filepath = asksaveasfilename(initialdir=initialdir, defaultextension=".fdmcal")

    with open(filepath, "wb") as cal_file:
        pickle.dump(calibration, cal_file)


def get_fdm_calibration():
    initialdir = get_config()["cal"]
    filepath = askopenfilename(initialdir=initialdir, filetypes=["SimpleCalibration {.fdmcal}"])

    with open(filepath, "rb") as cal_file:
        calibration = pickle.load(cal_file)

    return calibration


def save_calibration(calibration):
    if isinstance(calibration, SimpleCalibration):
        save_simple_calibration(calibration)
    elif isinstance(calibration, FDMCalibration):
        save_FDM_calibration(calibration)
