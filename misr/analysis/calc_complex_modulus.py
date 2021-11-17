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
    included = []
    cplx_Gs = []

    excluded = []
    excl_Gs = []
    for i, resp in enumerate(measured_responses):
        curr_G = (cal.tub.W/(4.0 * cal.rod.L)) * cal.alpha * (np.exp(-1.j*resp.rod_phase) / resp.AR -
                                                                            np.exp(-1.j*phase_func(resp.rod_freq)) / AR_func(resp.rod_freq))
        curr_Bo = AR_func(resp.rod_freq) / resp.AR

        if curr_Bo < 100:
            print("Excluded a result because of low Bo number!")
            excl_Gs.append(curr_G)
            excluded.append(measured_responses[i])
        else:
            cplx_Gs.append(curr_G)
            included.append(measured_responses[i])

    return FinalResults(cplx_Gs, included, cal, excluded, excl_Gs)


def calc_complex_modulus(cal, **kwargs):
    # MAY NOT BE WORKING CORRECTLY!!!!!
    # TODO: Incorporate ERRORS!!!!!!!!!!!!!!!!!!! - from here on out no errors were considered!!!!!!
    measured_responses = select_and_analyse(**kwargs)

    # Sort system responses by frequency
    responses = sorted(measured_responses, key=lambda sys_resp: sys_resp.rod_freq)

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in responses])

    # Num points
    N = 30

    max_p = np.log(tub.W/rod.d)
    eta = 1.0034e-3     # in [Pa*s] @ 20C --> TODO: PREBERI IZ FILA
    rho = 998.2         # in [kg/m^3] @ 20C --> TODO: PREBERI IZ FILA

    max_iter = 50

    Bo = []
    omegas = []
    included = []

    excluded = []
    Bo_excl = []
    omegas_excl = []
    for i, omega in enumerate([2*np.pi*resp.rod_freq for resp in responses]):
        Bo_curr = 100. + 0.j
        for iter_idx in range(max_iter):
            Re = rho * omega * ((rod.d / 2.) ** 2) / eta
            g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, Bo_curr, Re)

            calc_cplx_Foverz = (D_sub(g, omega, eta, hp, htheta, rod.L) +
                                D_surf(g, Bo_curr, omega, eta, hp, rod.L) +
                                cal.k - rod.m*omega*omega)

            Bo_change_factor = cal.alpha/(calc_cplx_Foverz * responses[i].AR * np.exp(1.j*responses[i].rod_phase))
            Bo_curr *= Bo_change_factor
            print(np.square(np.real(Bo_change_factor) - 1) + np.square(np.imag(Bo_change_factor)))

            if (np.square(np.real(Bo_change_factor) - 1) + np.square(np.imag(Bo_change_factor))) < np.square(0.05):
                print("Converged!\n")
                Bo.append(Bo_curr)
                omegas.append(omega)
                included.append(responses[i])
                break
        # If a break has occured in the for loop, the else will not compute.
        # If the loop hasn't converged, then else will execute!
        else:
            print("Didn't converge!\n")
            excluded.append(responses[i])
            Bo_excl.append(Bo_curr)
            omegas_excl.append(omega)

    Bo = np.array(Bo)
    omegas = np.array(omegas)

    Bo_excl = np.array(Bo_excl)
    omegas_excl = np.array(omegas_excl)

    # Pretvoriš iz Bo* v G
    cplx_Gs = omegas * rod.d/2 * eta * (np.real(Bo) - 1.j*np.imag(Bo))
    excl_cplx_Gs = omegas_excl * rod.d / 2 * eta * (np.real(Bo_excl) - 1.j * np.imag(Bo_excl))

    return FinalResults(cplx_Gs, included, cal, excluded, excl_cplx_Gs)
