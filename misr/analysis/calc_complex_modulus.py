import numpy as np
from .freq_and_phase_extract import select_and_analyse
from .ResultsClass import FinalResults
from .Rod_TubClass import get_Rod_and_Tub
from .num_sim_flowfield import D_surf, D_sub, flowfield_FDM


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


def calc_complex_modulus(cal, **kwargs):
    # MAY NOT BE WORKING CORRECTLY!!!!!
    # TODO: Incorporate ERRORS!!!!!!!!!!!!!!!!!!! - from here on out no errors were considered!!!!!!
    measured_responses = select_and_analyse(**kwargs)

    # Sort system responses by frequency
    responses = sorted(measured_responses, key=lambda sys_resp: sys_resp.rod_freq)

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in responses])

    # STARTING Bo
    Bo = np.ones(len(responses), dtype=np.complex128) * 10

    # Num points
    N = 30

    max_p = np.log(tub.W/rod.d)
    eta = 1.0034e-3     # in [Pa*s] @ 20C --> TODO: PREBERI IZ FILA
    rho = 998.2         # in [kg/m^3] @ 20C --> TODO: PREBERI IZ FILA

    omegas = np.array([resp.rod_freq*2*np.pi for resp in responses])

    Bo_change_factor = np.inf
    while np.all(np.abs(np.log10(np.abs(Bo_change_factor))) > 0.3):
        for i, omega in enumerate(omegas):
            Re = rho * omega * ((rod.d / 2.) ** 2) / eta
            g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, Bo[i], Re)

            calc_cplx_Foverz = (D_sub(g, omega, eta, hp, htheta, rod.L) +
                                D_surf(g, Bo[i], omega, eta, hp, rod.L) +
                                cal.k - rod.m*omega*omega)

            Bo_change_factor = cal.alpha/(calc_cplx_Foverz * responses[i].AR * np.exp(1.j*responses[i].rod_phase))
            Bo[i] *= Bo_change_factor
        print("Bo:", Bo)

    # Pretvoriš iz Bo* v G
    cplx_Gs = omegas * rod.d/2 * eta * (np.real(Bo) - 1.j*np.imag(Bo))

    return FinalResults(cplx_Gs, responses, cal)
