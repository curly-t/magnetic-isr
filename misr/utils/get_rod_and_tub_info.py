import numpy as np
from tkinter.filedialog import askopenfilename
    
def get_rod_info(initialdir=None):
    filepath = askopenfilename(initialdir=initialdir, title="Select Rod info file")
    # The fields are: (every value written in each row, all in one collumn)
    # id, L, m, d, L_err, m_err, d_err         (id, Length, mass, diameter)
    # Additional comments are saved in a comment
    return np.loadtxt(filepath)

def get_tub_info(initialdir=None):
    filepath = askopenfilename(initialdir=initialdir, title="Select Tub info file")
    # The fields are: (every value written in each row, all in one collumn)
    # id, W, h, V, W_err, h_err, V_err         (id, Width, hight, Volume (NOT total but water volume used  [SI units]))
    # Additional comments are saved in a comment
    return np.loadtxt(filepath)
