import math as m
from matplotlib.pyplot import get
import numpy as np
from ..utils.save_and_get_calibration import get_simple_calibration, save_simple_calibration
from ..utils.get_rod_and_tub_info import get_rod_info, get_tub_info
from .import_trackdata import select_filter_import_data
from .freq_and_phase_extract import freq_phase_ampl
from .calc_system_calibration import simple_calibration


def calc_simple_complex_modulus(Iampl="*", Ioffs="*", Ifreq="*", keyword_list=None, initialdir=None, freq_err=0.01, plot_track_results=False, complex_drift=True,
                                   rod_led_phase_correct=True, filter_for_wierd_phases=True, exceptable_phase_insanity=0.8*np.pi):
    # Use previous or calucalte a new calibration array --------------------------------------------------------------------
    cal_stat = str(input("Use previous calibration (type 'p') or calculate a new one (type 'n'): "))
    if cal_stat == "n":
        # TODO: MAKE OBJECTS FOR ROD, TUB and CALIBRATION!!!
        rod_info = get_rod_info(initialdir=initialdir)
        tub_info = get_tub_info(initialdir=initialdir)

        calibration_array = simple_calibration(initialdir=initialdir)
        save_cal = str(input("Save calibration? (Y or n): "))
        if save_cal == "Y":
            save_simple_calibration(calibration_array, rod_info, tub_info, initialdir=initialdir)
    
    elif cal_stat == "p":
        calibration_array, rod_info, tub_info = get_simple_calibration(initialdir=initialdir)
    
    else:
        print("No calibration array used - terminating!")
        exit()
    # ----------------------------------------------------------------------------------------------------------------------

    # TODO: Incorporate ERRORS!!!!!!!!!!!!!!!!!!! - from here on out no errors were considered!!!!!!
    # TODO: MAKE OBJECTS FOR THOSE THINGS!
    W = tub_info[1]
    L = rod_info[1]
    mass = rod_info[2]
    alpha = calibration_array[0]
    k = calibration_array[1]
    c = calibration_array[2]

    if keyword_list == None:
        keyword_list = []
    # THE SAME CODE AS IN calc_system_calibration.py
    # TODO: Make a function for this code! ------------- 10941
    measurements = select_filter_import_data(initialdir, Iampl, Ioffs, Ifreq, keyword_list)
    measured_responses = []
    for measrmnt in measurements:
        resp = freq_phase_ampl(measrmnt, freq_err, plot_track_results=plot_track_results, complex_drift=complex_drift,
                                   rod_led_phase_correct=rod_led_phase_correct, exceptable_phase_insanity=exceptable_phase_insanity)

        if filter_for_wierd_phases:
            if not resp.phase_sanity_check():
                measured_responses.append(resp)
            else:
                print("Measurement discarded because of insane phase!")
        else:
            measured_responses.append(resp)
    # --------------------------------------------------- 10941

    # THIST TWO FUNCTIONS BORROWED FROM calc_system_calibration.py
    # TODO: MAKE a functios accessible to this code ----------------------- 289475
    def AR_func(nu, c):
        return alpha/(np.sqrt((c * 2*np.pi*nu)**2 + (k - mass * 4.*np.pi*np.pi*nu*nu)**2))

    def phase_func(nu, c):
        return np.arctan2(-2*np.pi*nu * c, k - mass * (2*np.pi*nu)**2)
    # ----------------------------------------------------------------------- 289475

    # DIRECT LINEARNO ODŠTEJEŠ EFEKT SISTEMA OD EFEKTA PROTEINA!!!
    for resp in measured_responses:
        resp.complex_modulus = (W/(4.0 * L)) * calibration_array[0] * (np.exp(-1.j*resp.rod_phase) / resp.AR -
                                                                       np.exp(-1.j*phase_func(resp.rod_freq, c)) / AR_func(resp.rod_freq, c))

    # TODO: Dodaj opozorilo da to deluje le pri visokih Bousinesqovih številih (Bo > 100)

    return measured_responses
