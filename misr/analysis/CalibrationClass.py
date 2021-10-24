import time
from datetime import datetime


class SimpleCalibration:
    def __init__(self, cal_results, sys_responses, rod, tub):
        # This class is used to store the results of a simple calibration (aproximating 2nd order system)
        self.alpha = cal_results[0]
        self.k = cal_results[1]
        self.c = cal_results[2]
        self.alpha_err = cal_results[3]
        self.k_err = cal_results[4]
        self.c_err = cal_results[5]

        self.sys_resps = sys_responses

        self.timestamp = time.time()

        self.rod = rod
        self.tub = tub

    def date(self):
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')

    def dirnames(self):
        return [sr.meas.dirname for sr in self.sys_resps]


class FDMCalibration:
    def __init__(self, cal_results, sys_responses, rod, tub):
        self.alpha = cal_results[0]
        self.k = cal_results[1]
        # TODO:
        # self.alpha_err = cal_results[2]
        # self.k_err = cal_results[3]

        self.sys_resps = sys_responses

        self.timestamp = time.time()

        self.rod = rod
        self.tub = tub

    def date(self):
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')

    def dirnames(self):
        return [sr.meas.dirname for sr in self.sys_resps]
