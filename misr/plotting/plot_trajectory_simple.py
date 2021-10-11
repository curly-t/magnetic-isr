from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# LOAD A FILE
df = np.loadtxt('offs900_ampl800_freq3857.dat')

plt.plot(df[:,1], df[:,2], '.')
plt.show()