import numpy as np
from scipy.optimize import leastsq

from .freq_and_phase_extract import select_and_analyse
from .ResultsClass import FinalResults
from .Rod_TubClass import get_Rod_and_Tub
from .num_sim_flowfield import D_surf, D_sub, flowfield_FDM
from gvar import mean as gvalue
from gvar import gvar, evalcov


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
        def min_func(Bo_curr, return_floats=True):
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

            if return_floats:
                return gvalue(Bo_change_factor) - np.array([1, 0])
            else:
                return Bo_curr * (Bo_change_factor - np.array([1, 0]))

        best_Bo, best_Bo_unscal_cov, infodict, message, ier = leastsq(min_func, np.array([100, 0]), full_output=True)
        best_Bo_cov = best_Bo_unscal_cov * np.var(min_func(best_Bo))

        # Potem pa moramo dodati še kovariančno matriko ki pride iz napake izračuna samega factorja!
        cov_from_factor_eval = evalcov(min_func(best_Bo, return_floats=False))

        # TODO: FIGURE OUT HOW TO PROPAGATE THE ERROR TROUGH THE PROGRAM WITHOUT MID-PROGRAM DEFINITIONS like
        # bo.append(gvar(best_Bo, total_Bo_cov)) because ths the prohibits the extraction of correlation to the
        # primitive inputs!!! (like how does rod length impact the final error, or how much does I_ampl matter???
        # TODO: FIGURE OUT HOW TO PROPAGATE ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        total_Bo_cov = best_Bo_cov + cov_from_factor_eval
        print(f"cov_from_min:\n{best_Bo_cov}\ncov_from_factor_eval:\n{cov_from_factor_eval}\n")

        if ier in [1, 2, 3, 4]:
            print("Success!")
            Bo.append(gvar(best_Bo, total_Bo_cov))
            omegas.append(omega)
            included.append(responses[i])
        else:
            print(message)
            Bo_excl.append(gvar(best_Bo, total_Bo_cov))
            omegas_excl.append(omega)
            excluded.append(responses[i])

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
