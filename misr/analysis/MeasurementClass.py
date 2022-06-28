import numpy as np
import os
from datetime import datetime
import re
from gvar import gvar
from ..utils.get_rod_and_tub_info import guess_rod_and_tub


class Measurement:
    def __init__(self, filepath, filter_duplicates=True, max_num_points=300, guess_rod_orient=True):
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

        self.import_measurement_config(guess_rod_orient)

        trackData = np.loadtxt(filepath)
        if filter_duplicates:
            non_duplicate_idxs = self.filter_duplicate_data(trackData)
            trackData = trackData[non_duplicate_idxs]

        if max_num_points is not None:
            selected_points = self.dilute_number_of_points(len(trackData), max_num_points)
            trackData = trackData[selected_points]

        # The trackData array is comprised of 4 columns:
        # frameIdx, frameTime, rodEdgePos, brightness
        self.timeLength = trackData[-1, 1] - trackData[0, 1]
        self.numFrames = len(trackData)
        self.frameIdxs = trackData[:, 0]                                            # There can be no error in index
        self.times = gvar(trackData[:, 1], np.ones(self.numFrames)*0.0005)          # Assuming 0.5 ms error in time
        self.positions = gvar(trackData[:, 2], np.ones(self.numFrames)*0.5)         # Assuming half pixel of resolution
        self.brights = gvar(trackData[:, 3], np.ones(self.numFrames)/np.sqrt(100))  # Assuming calc by 10x10 average

    def date_of_last_mod(self):
        return datetime.fromtimestamp(self.timestamp_of_last_mod).strftime('%Y-%m-%d %H:%M:%S')

    def import_measurement_config(self, guess_rod_orient):
        full_folder_path = os.path.split(self.filepath)[0]
        meas_config_path = os.path.join(full_folder_path, ".meas_config")
        if os.path.isfile(meas_config_path):
            with open(meas_config_path, "r") as meas_config_file:
                file_contents = "\n".join(meas_config_file.readlines())
                self.x_pixels = int(re.search("X_PIXEL_COUNT='.*'", file_contents)[0][15:-1])
                self.pixel_size = float(re.search("PIXEL_SIZE='.*'", file_contents)[0][12:-1])
                self.rod_id = int(re.search("ROD_ID='.*'", file_contents)[0][8:-1])
                self.tub_id = int(re.search("TUB_ID='.*'", file_contents)[0][8:-1])
                self.rod_orient = re.search("ROD_ORIENT='.*'", file_contents)
                # Some measurements will have no information about rod orientation
            if self.rod_orient is not None:
                self.rod_orient = int(self.rod_orient[0][12:-1])
            else:
                if not guess_rod_orient:
                    self.rod_orient = 1
                else:
                    # THIS IMPORT MUST BE HERE TO AVOID CIRCULAR IMPORTS
                    from ..utils.guess_rod_orientation import guess_rod_orientation
                    from ..utils.measurement_utils import make_measurement_config_file
                    print("ROD ORIENTATION FOR THIS FILE NOT KNOWN!")
                    print("Guessing rod orientation and making new meas_config file!")
                    self.rod_orient = guess_rod_orientation(full_folder_path)
                    make_measurement_config_file(full_folder_path, self.rod_id, self.tub_id,
                                                 self.rod_orient, self.x_pixels, self.pixel_size)

        else:
            # THIS IMPORT MUST BE HERE TO AVOID CIRCULAR IMPORTS
            from ..utils.guess_rod_orientation import guess_rod_orientation
            print("IMPORTING MEASUREMENT WITHOUT MEASUREMENT CONFIG FILE!")
            print("DEFAULT MODE 600 x 960 pixels ASSUMED!!!!!!")
            self.x_pixels = 960
            self.pixel_size = 0.00000296      # DEFAULT VALUE FOR 600x960
            self.rod_id, self.tub_id = guess_rod_and_tub([self.dirname])    # If nothing --> rod_id, tub_id = None, None
            if not guess_rod_orient:
                self.rod_orient = 1
            else:
                print("Guessing the rod orientation!")
                self.rod_orient = guess_rod_orientation(full_folder_path)
                # Ne moreš nardit novega configa ker ne veš kakšna sta bla rod in tub!


    def filter_duplicate_data(self, trackdata):
        non_duplicates = np.where(trackdata[:-1, 1] != trackdata[1:, 1])[0]
        if trackdata[-1, 1] != trackdata[-2, 1]:
            non_duplicates = np.array(list(non_duplicates) + [len(trackdata) - 1])
        return non_duplicates

    def dilute_number_of_points(self, num_points, max_num):
        if num_points > max_num:
            selected_indices = np.int0(np.linspace(0, num_points-1, num=max_num))
            return selected_indices
        else:
            return np.arange(num_points)


# Če bo potrebno kasneje devat vse iz iste mape v measurement run! :)
class MeasurementRun():
    def __init__(self, measurements):
        self.measurements = measurements
        # PREDVIDEVAMO DA SO VSI PODATKI IZ MeasurementRun IZ ISTE MAPE!!!
        self.dirname = measurements[0].dirname
        self.timestamp_of_last_mod = min([m.timestamp_of_last_mod for m in measurements])
    
    def date_of_last_mod(self):
        return datetime.fromtimestamp(self.timestamp_of_last_mod).strftime('%Y-%m-%d %H:%M:%S')

