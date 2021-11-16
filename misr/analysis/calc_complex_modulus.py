import numpy as np
from .freq_and_phase_extract import select_and_analyse
from .ResultsClass import FinalResults


def calc_simple_complex_modulus(cal, **kwargs):
    # TODO: Incorporate ERRORS!!!!!!!!!!!!!!!!!!! - from here on out no errors were considered!!!!!!

    measured_responses = select_and_analyse(**kwargs)

    def AR_func(nu):
        return cal.alpha/(np.sqrt((cal.c * 2*np.pi*nu)**2 + (cal.k - cal.rod.m * 4.*np.pi*np.pi*nu*nu)**2))

    def phase_func(nu):
        return np.arctan2(-2*np.pi*nu * cal.c, cal.k - cal.rod.m * (2*np.pi*nu)**2)

    # DIRECT LINEARNO ODŠTEJEŠ EFEKT SISTEMA OD EFEKTA PROTEINA - SIMPLEST CASE
    cplx_Gs = np.zeros((len(measured_responses,)), dtype=np.complex128)
    Bo = np.zeros((len(measured_responses),))
    excld_ans = ""
    for i, resp in enumerate(measured_responses):
        cplx_Gs[i] = (cal.tub.W/(4.0 * cal.rod.L)) * cal.alpha * (np.exp(-1.j*resp.rod_phase) / resp.AR -
                                                                            np.exp(-1.j*phase_func(resp.rod_freq)) / AR_func(resp.rod_freq))
        Bo[i] = AR_func(resp.rod_freq) / resp.AR

        if Bo[-1] < 100 and excld_ans == "":
            print("WARNING - Results may be inconclusive as the Bo < 100 sometimes!")
            excld_ans = str(input("Exclude inconclusive results? [Y/n] "))
            if excld_ans == "Y":
                continue

    return FinalResults(cplx_Gs, measured_responses, cal)
