import numpy as np
import os
from datetime import datetime
import re
from gvar import gvar


class Measurement:
    def __init__(self, filepath):
        # Information about the filepath, filename and filedir
        self.filepath = filepath
        self.filename = os.path.split(filepath)[1]
        self.dirname = os.path.split(os.path.split(filepath)[0])[1]
        # Timestamp of last change (in s)
        self.timestamp_of_last_mod = os.path.getmtime(filepath)
        # Values for Ifreq, Ioffset and Iamplitude, represented in the program as Hz, and A, so as base SI units
        self.Ifreq = gvar(int((re.search(r"freq\d+", filepath)[0])[4:])/1000, 0.001)
        self.Ioffs = gvar(int((re.search(r"offs\d+", filepath)[0])[4:])/1000, 0.0008)
        self.Iampl = gvar(int((re.search(r"ampl\d+", filepath)[0])[4:])/1000, 0.0008)
        # FIND / MEASURE ERROR VALUES!!! IT IS OF EXTREME IMPORTANCE TO THE ERROR ESTIMATION!!!
        # NATAN ANSWER --> cca 0.8mA of error combined!   -- tu dal pol napake Ioffs in pol Iampl

        # Info on the pixel size calibration:
        # Pixel size determines the size in real life, of one pixel on the screen.
        # Used for conversion from pixel coordinates to [m]
        # pixel_size = m/pixel
        self.pixel_size = 0.00005   # Measured pixel size

        # The trackData array is comprised of 4 columns:
        # frameIdx, frameTime, rodEdgePos, brightness
        trackData = np.loadtxt(filepath)
        self.timeLength = trackData[-1, 1] - trackData[0, 1]
        self.numFrames = len(trackData)

        self.frameIdxs = trackData[:, 0]                                            # There can be no error in index
        self.times = gvar(trackData[:, 1], np.ones(self.numFrames)*0.0005)          # Assuming 0.5 ms error in time
        self.positions = gvar(trackData[:, 2], np.ones(self.numFrames)*0.5)         # Assuming half pixel of resolution
        self.brights = gvar(trackData[:, 3], np.ones(self.numFrames)/np.sqrt(100))  # Assuming calc by 10x10 average

    def date_of_last_mod(self):
        return datetime.fromtimestamp(self.timestamp_of_last_mod).strftime('%Y-%m-%d %H:%M:%S')


# Če bo potrebno kasneje devat vse iz iste mape v measurement run! :)
class MeasurementRun():
    def __init__(self, measurements):
        self.measurements = measurements
        # PREDVIDEVAMO DA SO VSI PODATKI IZ MeasurementRun IZ ISTE MAPE!!!
        self.dirname = measurements[0].dirname
        self.timestamp_of_last_mod = min([m.timestamp_of_last_mod for m in measurements])
    
    def date_of_last_mod(self):
        return datetime.fromtimestamp(self.timestamp_of_last_mod).strftime('%Y-%m-%d %H:%M:%S')

