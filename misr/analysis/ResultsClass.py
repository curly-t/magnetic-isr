from numpy import pi as np_pi
from numpy import array as np_array
from gvar import mean as gvalue


class SingleResult:
    def __init__(self, result_dict, measurement_object, mean_rod_position):
        # This class is used to store the results from a single measurement
        # (one date, one run, one frequency - not all frequencies of the measurement run)

        # Measurement object is linked to this single result object
        self.meas = measurement_object

        # Values for rodfreq, rodampl, rodphase
        self.rod_freq = result_dict["rod_freq"]             # This in nu - actual freq, not omega
        self.rod_ampl = result_dict["rod_ampl"]
        self.rod_phase = result_dict["rod_phase"]
        self.rod_mean = gvalue(mean_rod_position)

        # Calculate AR (amplitude ratio between rod_ampl and Iampl)
        self.AR = self.meas.pixel_size * self.rod_ampl / self.meas.Iampl

    def phase_sanity_check(self):
        # Returns False (0) if there is no error, and True (1) if there is a phase error
        return not (0 > gvalue(self.rod_phase) > -np_pi)


class FinalResults:
    def __init__(self, cplx_Gs, single_results, calibration, excluded=None, excluded_cplx_Gs=None):
        self.G = cplx_Gs
        self.SRs = single_results
        self.cal = calibration
        self.freqs = np_array([resp.rod_freq for resp in self.SRs])

        if excluded is None:
            self.excl = []
            self.excl_freqs = np_array([])
        else:
            self.excl = excluded
            self.excl_freqs = np_array([resp.rod_freq for resp in self.excl])

        if excluded_cplx_Gs is None:
            self.excl_G = []
        else:
            self.excl_G = excluded_cplx_Gs



