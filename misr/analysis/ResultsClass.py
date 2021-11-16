from numpy import pi as np_pi


class SingleResult:
    def __init__(self, result_dict, measurement_object, exceptable_phase_insanity):
        # This class is used to store the results from a single measurement
        # (one date, one run, one frequency - not all frequencies of the measurement run)

        # Measurement object is linked to this single result object
        self.meas = measurement_object

        # Values for rodfreq, rodampl, rodphase
        self.rod_freq = result_dict["rod_freq"]             # This in nu - actual freq, not omega
        self.rod_freq_err = result_dict["rod_freq_err"]
        self.rod_ampl = result_dict["rod_ampl"]
        self.rod_ampl_err = result_dict["rod_ampl_err"]
        self.rod_phase = result_dict["rod_phase"]
        self.rod_phase_err = result_dict["rod_phase_err"]
        self.exceptable_phase_insanity = exceptable_phase_insanity

        # Calculate AR (amplitude ratio between rod_ampl and Iampl)
        self.AR = self.meas.pixel_size * self.rod_ampl / self.meas.Iampl
        # AR is a ratio, so the error is calculated as the sum of two relative errors
        self.AR_err = (self.rod_ampl_err/self.rod_ampl + self.meas.Iampl_err/self.meas.Iampl) * self.AR

    def phase_sanity_check(self):
        # Returns False (0) if there is no error, and True (1) if there is a phase error
        return not (self.exceptable_phase_insanity > self.rod_phase > (- np_pi - self.exceptable_phase_insanity))


class FinalResults:
    def __init__(self, cplx_Gs, single_results, calibration):
        self.G = cplx_Gs
        self.SRs = single_results
        self.cal = calibration
