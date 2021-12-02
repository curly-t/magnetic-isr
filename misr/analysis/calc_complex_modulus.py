import numpy as np
from scipy.optimize import root

from .freq_and_phase_extract import select_and_analyse
from .ResultsClass import FinalResults
from .Rod_TubClass import get_Rod_and_Tub
from .num_sim_flowfield import D_surf, D_sub, flowfield_FDM
from gvar import mean as gvalue
from gvar import gvar


def calc_simple_complex_modulus(cal, **kwargs):
    measured_responses = select_and_analyse(**kwargs)

    def AR_func(nu):
        return cal.alpha/(np.sqrt((cal.c * 2*np.pi*nu)**2 + (cal.k - cal.rod.m * 4.*np.pi*np.pi*nu*nu)**2))

    def phase_func(nu):
        return np.arctan2(-2*np.pi*nu * cal.c, cal.k - cal.rod.m * (2*np.pi*nu)**2)

    # Complex Gs are now stored as arrays of length 2: cplx_Gs=[np.array([G1_Re, G1_Im]), np.array([G2_Re, G2_Im]), ...]
    included = []
    cplx_Gs = []

    excluded = []
    excl_Gs = []
    for i, resp in enumerate(measured_responses):
        # DIRECT LINEARNO ODŠTEJEŠ EFEKT SISTEMA OD EFEKTA PROTEINA - SIMPLEST CASE
        curr_G = (cal.tub.W/(4.0 * cal.rod.L)) * cal.alpha * \
                 (np.array([np.cos(-resp.rod_phase), np.sin(-resp.rod_phase)]) / resp.AR -
                  np.array([np.cos(-phase_func(resp.rod_freq)), np.sin(-phase_func(resp.rod_freq))]) / AR_func(resp.rod_freq))

        curr_real_bo = AR_func(resp.rod_freq) / resp.AR

        if curr_real_bo < 100:
            print("Excluded a result because of low Bo number!")
            excl_Gs.append(curr_G)
            excluded.append(measured_responses[i])
        else:
            cplx_Gs.append(curr_G)
            included.append(measured_responses[i])

    return FinalResults(np.array(cplx_Gs), included, cal, excluded, np.array(excl_Gs))


def calc_complex_modulus(cal, **kwargs):
    measured_responses = select_and_analyse(**kwargs)

    # Sort system responses by frequency
    responses = sorted(measured_responses, key=lambda sys_resp: sys_resp.rod_freq)

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in responses])

    # Num points
    N = 30

    max_p = np.log(gvalue(tub.W/rod.d))
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
        def min_func(Bo_curr):
            Re = gvalue(rho * omega * ((rod.d / 2.) ** 2) / eta)
            g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, Bo_curr[0] + 1.j*Bo_curr[1], Re)

            cplx_Fovz = (D_sub(g, omega, eta, hp, htheta, rod.L) +
                         D_surf(g, Bo_curr, omega, eta, hp, rod.L) +
                         cal.k - rod.m*omega*omega)

            # Calculating Bo change factor
            phi = responses[i].rod_phase
            real_factor = cal.alpha/(responses[i].AR * (np.square(cplx_Fovz[0]*np.cos(phi) - cplx_Fovz[1]*np.sin(phi)) +
                                                        np.square(cplx_Fovz[1]*np.cos(phi) + cplx_Fovz[0]*np.sin(phi))))
            Bo_change_factor = np.array([cplx_Fovz[0]*np.cos(phi) - cplx_Fovz[1]*np.sin(phi),
                                         cplx_Fovz[1]*np.cos(phi) + cplx_Fovz[0]*np.sin(phi)]) * real_factor

            # Dodaj MOŽNO NAPAKO ZNOTRAJ BO CHANGE FACTORJA!!! (PRETVORBA V gvalue() JE NUJNA ZA DELOVANJE,
            # AMPAK ZA OCENO NAPAKE PA NI VREDU, DODAJ NA KONCU!!
            return np.square(gvalue(Bo_change_factor) - np.array([1, 0]))

        # DODAJ ZAUSTAVLJALNI POGOJ DA GLEDA LE NA VELIKOST FUNKCIJE IN NE NA GRADIENT ZA ZAUSTAVLJANJE
        min_results = root(min_func, np.array([100, 0]), method="lm")
        # MOGOČE TU ŠE ENKRAT ZAŽENEŠ DA POGRUNTAŠ NAPAKO!? (MOGOČE ZAŽENEŠ ŠE ENKRAT S TEM DA MALO
        # SPREMENIŠ OBJEKTNO FUNKCIJO, DA TI DAJE V SKALAR, IN POTEM UPORABIŠ VGRAJENO METODO MINIMIZE KI TI
        # ZA SKALARJE NAJBRŽ VRNE TUDI HESSOVO MATRIKO.

        if min_results.success:
            print("Success!")
            # Dodaj MOŽNO NAPAKO ZNOTRAJ BO CHANGE FACTORJA!!!
            Bo.append(gvar(min_results.x, np.zeros_like(min_results.x)))
            omegas.append(omega)
            included.append(responses[i])
        else:
            print(min_results.message)
            Bo_excl.append(gvar(min_results.x, np.zeros_like(min_results.x)))
            omegas_excl.append(omega)
            excluded.append(responses[i])

        # Bo_curr = np.array([gvar(100, 1), gvar(1, 1)])
        # for iter_idx in range(max_iter):
        #     Re = gvalue(rho * omega * ((rod.d / 2.) ** 2) / eta)
        #     g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, gvalue(Bo_curr[0]) + 1.j*gvalue(Bo_curr[1]), Re)
        #
        #     cplx_Fovz = (D_sub(g, omega, eta, hp, htheta, rod.L) +
        #                  D_surf(g, Bo_curr, omega, eta, hp, rod.L) +
        #                  cal.k - rod.m*omega*omega)
        #
        #     # Calculating Bo change factor
        #     phi = responses[i].rod_phase
        #     real_factor = cal.alpha/(responses[i].AR * (np.square(cplx_Fovz[0]*np.cos(phi) - cplx_Fovz[1]*np.sin(phi)) +
        #                                                 np.square(cplx_Fovz[1]*np.cos(phi) + cplx_Fovz[0]*np.sin(phi))))
        #     Bo_change_factor = np.array([cplx_Fovz[0]*np.cos(phi) - cplx_Fovz[1]*np.sin(phi),
        #                                  cplx_Fovz[1]*np.cos(phi) + cplx_Fovz[0]*np.sin(phi)]) * real_factor
        #     Bo_curr *= Bo_change_factor
        #     print(np.square(Bo_change_factor[0] - 1) + np.square(Bo_change_factor[1]))
        #
        #     if (np.square(Bo_change_factor[0] - 1) + np.square(Bo_change_factor[1])) < np.square(0.05):
        #         print("Converged!\n")
        #         Bo.append(Bo_curr)
        #         omegas.append(omega)
        #         included.append(responses[i])
        #         break
        # # If a break has occured in the for loop, the else will not compute.
        # # If the loop hasn't converged, then else will execute!
        # else:
        #     print("Didn't converge!\n")
        #     excluded.append(responses[i])
        #     Bo_excl.append(Bo_curr)
        #     omegas_excl.append(omega)

    Bo = np.array(Bo)
    omegas = np.array(omegas)

    Bo_excl = np.array(Bo_excl)
    omegas_excl = np.array(omegas_excl)

    # Pretvoriš iz Bo* v G
    if len(Bo) > 0:
        cplx_Gs = rod.d/2 * eta * np.transpose(np.array([Bo[:, 0] * omegas,
                                                         -Bo[:, 1] * omegas]))
    else:
        cplx_Gs = np.array([])

    # Isto za izključene neskonvergirane podatke
    if len(Bo_excl) > 0:
        excl_cplx_Gs = omegas_excl * rod.d/2 * eta * np.transpose(np.array([Bo_excl[:, 0] * omegas_excl,
                                                                            -Bo_excl[:, 1] * omegas_excl]))
    else:
        excl_cplx_Gs = np.array([])

    return FinalResults(cplx_Gs, included, cal, excluded, excl_cplx_Gs)
