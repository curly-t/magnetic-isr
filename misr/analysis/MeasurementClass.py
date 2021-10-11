import numpy as np
import os
from datetime import datetime
import re


# Če bi slučajno še kdaj potrebovali file z decimalnimi vejicami namesto pikami dekodirat!
# def number_with_comma_to_float(column):
#     return bytes(column.decode("utf-8").replace(",", "."), "utf-8")

class Measurement():
    def __init__(self, filepath):
        # Information about the filepath, filename and filedir
        self.filepath = filepath
        self.filename = os.path.split(filepath)[1]
        self.dirname = os.path.split(os.path.split(filepath)[0])[1]
        # Timestamp of last change (in s)
        self.timestamp_of_last_mod = os.path.getmtime(filepath)
        # Values for Ifreq, Ioffset and Iamplitude
        self.Ifreq = int((re.search(r"freq\d+", filepath)[0])[4:])/1000
        self.Ioffs = int((re.search(r"offs\d+", filepath)[0])[4:])/1000
        self.Iampl = int((re.search(r"ampl\d+", filepath)[0])[4:])/1000
        # Values for rodfreq, rodampl, rodphase. Initially uninitialized.
        self.rod_freq = None
        self.rod_freq_err = None
        self.rod_ampl = None
        self.rod_ampl_err = None
        self.rod_phase = None
        self.rod_pahse_err = None

        # The trackData array is comprised of 4 columns:
        # frameIdx, frameTime, rodEdgePos, brightness
        self.trackData = np.loadtxt(filepath)
        self.frameIdxs = self.trackData[:, 0]
        self.times = self.trackData[:, 1]
        self.positions = self.trackData[:, 2]
        self.brights = self.trackData[:, 3]

        self.timeLength = self.trackData[-1, 1] - self.trackData[0, 1]
        self.numFrames = len(self.trackData)

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

