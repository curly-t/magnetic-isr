import numpy as np
from tkinter.filedialog import asksaveasfilename, askopenfilename

def save_simple_calibration(calibration_arr, initialdir=None):
    filepath = asksaveasfilename(initialdir=initialdir)
    # The fields are:
    # alpha, k, c, alpha_err, k_err, c_err
    np.savetxt(filepath, calibration_arr)
    
def get_simple_calibration(initialdir=None):
    filepath = askopenfilename(initialdir=initialdir)
    # The fields are:
    # alpha, k, c, alpha_err, k_err, c_err
    return np.loadtxt(filepath)
    