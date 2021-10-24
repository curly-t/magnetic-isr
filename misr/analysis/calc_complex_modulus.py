import numpy as np
from ..utils.save_and_get_calibration import get_simple_calibration
from .freq_and_phase_extract import select_and_analyse
from .calc_system_calibration import simple_calibration


def calc_simple_complex_modulus(**kwargs):
    cal_stat = str(input("Use previous calibration (type 'p') or calculate a new one (type 'n'): "))
    if cal_stat == "n":
        cal = simple_calibration(**kwargs)
    
    elif cal_stat == "p":
        cal = get_simple_calibration()

    # TODO: Incorporate ERRORS!!!!!!!!!!!!!!!!!!! - from here on out no errors were considered!!!!!!

    measured_responses = select_and_analyse(**kwargs)

    def AR_func(nu):
        return cal.alpha/(np.sqrt((cal.c * 2*np.pi*nu)**2 + (cal.k - cal.rod.m * 4.*np.pi*np.pi*nu*nu)**2))

    def phase_func(nu):
        return np.arctan2(-2*np.pi*nu * cal.c, cal.k - cal.m * (2*np.pi*nu)**2)

    # DIRECT LINEARNO ODŠTEJEŠ EFEKT SISTEMA OD EFEKTA PROTEINA - SIMPLEST CASE
    for resp in measured_responses:
        resp.complex_modulus = (cal.tub.W/(4.0 * cal.rod.L)) * cal.alpha * (np.exp(-1.j*resp.rod_phase) / resp.AR -
                                                                            np.exp(-1.j*phase_func(resp.rod_freq)) / AR_func(resp.rod_freq))

    # TODO: Dodaj opozorilo da to deluje le pri visokih Bousinesqovih številih (Bo > 100)

    return measured_responses
