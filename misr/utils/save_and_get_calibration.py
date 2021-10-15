import numpy as np
from tkinter.filedialog import asksaveasfilename, askopenfilename

def save_simple_calibration(calibration_arr, rod_info, tub_info, initialdir=None):
    filepath = asksaveasfilename(initialdir=initialdir)
    # The fields are:
    # alpha, k, c, alpha_err, k_err, c_err, rod_id, rod_L, rod_m, rod_d, rod_L_err, rod_m_err, rod_d_err, tub_id, tub_W, tub_h, tub_V, tub_W_err, tub_h_err, tub_V_err
    # First six values are calibration constants, then the rod: id, Length, mass, diameter (and errors), then tub: id, Width, total height, water volume (and errors) [SI units]
    np.savetxt(filepath, np.concatenate([calibration_arr, rod_info, tub_info]))
    
def get_simple_calibration(initialdir=None):
    filepath = askopenfilename(initialdir=initialdir)
    # The fields are:
    # alpha, k, c, alpha_err, k_err, c_err, rod_id, rod_L, rod_m, rod_d, rod_L_err, rod_m_err, rod_d_err, tub_id, tub_W, tub_h, tub_V, tub_W_err, tub_h_err, tub_V_err
    # First six values are calibration constants, then the rod: id, Length, mass, diameter (and errors), then tub: id, Width, total height, water volume (and errors) [SI units]
    cal_info = np.loadtxt(filepath)
    calibration_array = cal_info[:6]
    rod_info = cal_info[6:13]
    tub_info = cal_info[13:]
    return calibration_array, rod_info, tub_info
