import numpy as np
import os
from datetime import datetime
import re
from gvar import gvar
from ..utils.get_rod_and_tub_info import guess_rod_and_tub


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

        self.import_measurement_config()

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

    def import_measurement_config(self):
        full_folder_path = os.path.split(self.filepath)[0]
        meas_config_path = os.path.join(full_folder_path, ".meas_config")
        if os.path.isfile(meas_config_path):
            with open(meas_config_path, "r") as meas_config_file:
                file_contents = "\n".join(meas_config_file.readlines())
                self.x_pixels = int(re.search("X_PIXEL_COUNT='.*'", file_contents)[0][15:-1])
                self.pixel_size = float(re.search("PIXEL_SIZE='.*'", file_contents)[0][12:-1])
                self.rod_id = int(re.search("ROD_ID='.*'", file_contents)[0][8:-1])
                self.tub_id = int(re.search("TUB_ID='.*'", file_contents)[0][8:-1])
        else:
            print("IMPORTING MEASUREMENT WITHOUT MEASUREMENT CONFIG FILE!")
            print("DEFAULT MODE 600 x 960 pixels ASSUMED!!!!!!")
            self.x_pixels = 960
            self.pixel_size = 0.00000296      # DEFAULT VALUE FOR 600x960
            self.rod_id, self.tub_id = guess_rod_and_tub([self.dirname])
            # If found nothing - returned rod and tub ids will be None


# Če bo potrebno kasneje devat vse iz iste mape v measurement run! :)
class MeasurementRun():
    def __init__(self, measurements):
        self.measurements = measurements
        # PREDVIDEVAMO DA SO VSI PODATKI IZ MeasurementRun IZ ISTE MAPE!!!
        self.dirname = measurements[0].dirname
        self.timestamp_of_last_mod = min([m.timestamp_of_last_mod for m in measurements])
    
    def date_of_last_mod(self):
        return datetime.fromtimestamp(self.timestamp_of_last_mod).strftime('%Y-%m-%d %H:%M:%S')

