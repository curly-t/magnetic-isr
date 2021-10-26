import enum
import math as m
from os import system
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit, minimize

from .freq_and_phase_extract import select_and_analyse

from .num_sim_flowfield import flowfield_FDM, D_sub
from .Rod_TubClass import get_Rod_and_Tub

from .CalibrationClass import SimpleCalibration, FDMCalibration
from ..utils.save_and_get_calibration import save_calibration

from ..plotting.plot_system_responses import plot_sr


def estimate_freq_range_borders(system_responses):
    """
        Estimates the frequency range borders for low frequency plateau, and for the high frequencies inertial damping.
        Because this needs only be done ONCE for a single magnetic rod, this is done manually.

        Lower limit is set: key "h" + left mouse click
        Upper limit is set: key "j" + left mouse click

        When done, close figure with key "q"
    """
    print("Drži 'h' in klikni z miško da izbereš nizkofrekvenčni plato.")
    print("Drži 'j' in klikni z miško da izbereš visokofrekvenčni plato.")
    print("Ko končaš, zapri figuro.")

    fig, ax_AR, ax_phase = plot_sr(system_responses)

    def modify_low_border(set_value):
        global low_border
        low_border = set_value
        print("Low border saved at log(freq) = {0}".format(low_border))
    
    def modify_high_border(set_value):
        global high_border
        high_border = set_value
        print("High border saved at log(freq) = {0}".format(high_border))

    def construct_event_handler(fig, ax_AR, ax_phase):
        AR_vline_low = ax_AR.axvline(-0.5, color='k')
        AR_vline_high = ax_AR.axvline(0.5, color='k')
        phase_vline_low = ax_phase.axvline(-0.5, color='k')
        phase_vline_high = ax_phase.axvline(0.5, color='k')
        def on_mouse_click(event):
            if event.xdata != None:
                if event.key == 'h':
                    # Low border
                    AR_vline_low.set_xdata(event.xdata)
                    phase_vline_low.set_xdata(event.xdata)
                    modify_low_border(event.xdata)
                elif event.key == 'j':
                    # High border
                    AR_vline_high.set_xdata(event.xdata)
                    phase_vline_high.set_xdata(event.xdata)
                    modify_high_border(event.xdata)
            # Redraw!
            fig.canvas.draw()
        return on_mouse_click

    on_mouse_click = construct_event_handler(fig, ax_AR, ax_phase)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    plt.show()

    # Ko zapreš figure, je izbira mej končana. Program izračuna točke, ki so vključene v posamezen režim
    low_idx_border = 0
    high_idx_border = len(system_responses) - 1

    for i in range(len(system_responses) - 1):
        if m.log10(system_responses[i + 1].rod_freq) >= low_border > m.log10(system_responses[i].rod_freq):
            low_idx_border = i + 1
        elif m.log10(system_responses[i + 1].rod_freq) >= high_border > m.log10(system_responses[i].rod_freq):
            high_idx_border = i + 1
    
    return low_idx_border, high_idx_border


def calculate_force_current_factor(high_freq_system_responses, mass, mass_err, normal_power_tol=1):
    # Construct arrays for line fitting
    ARs = [sys_resp.AR for sys_resp in high_freq_system_responses]
    AR_errs = [sys_resp.AR_err for sys_resp in high_freq_system_responses]
    freqs = [sys_resp.rod_freq for sys_resp in high_freq_system_responses]

    # Fit a line trough the high frequency regime data points
    # Error is returned unscaled, wights must satisfy: weights = 1/sigma**2
    high_freq_AR_coefs, high_freq_AR_cov = np.polyfit(np.log10(freqs), np.log10(ARs),
                                                      deg=1,
                                                      w=np.square(m.log(10) * np.array(ARs) / np.array(AR_errs)),
                                                      cov='unscaled')
    if (high_freq_AR_coefs[0] + 2)/m.sqrt(high_freq_AR_cov[0, 0]) >= normal_power_tol:
        warnings.warn("Potencna odvisnost omege pri visokih frekvencah je {0:2.3f} ({1:2.3e})!".format(high_freq_AR_coefs[0], m.sqrt(high_freq_AR_cov[0, 0])))
    
    # Intermediate constant, A = alpha/(4*m*pi**2)
    A = m.pow(10, high_freq_AR_coefs[1])
    
    # Final current to force converting constant alpha, a property of the rod. Alpha [N/A]
    alpha = 4. * (np.pi ** 2) * mass * A

    # Intermediate error calculation step
    A_err = m.log(10) * A * m.sqrt(high_freq_AR_cov[1, 1])

    # Final error for current to force converting constant alpha.
    alpha_err = (mass_err/mass + A_err/A) * alpha

    return alpha, alpha_err


def calculate_system_compliance(low_freq_system_responses, alpha, alpha_err):
    ARs = [sys_resp.AR for sys_resp in low_freq_system_responses]
    AR_errs = [sys_resp.AR_err for sys_resp in low_freq_system_responses]

    # Optimal combination of ARs is to weight them by weights = 1./sigma**2 (Wikipedia)
    AR_avg = np.average(ARs, weights=1./np.square(AR_errs))
    # Wikipedia: sigma_wighted_average_AR = 1 / (sqrt(sum(weights)))
    AR_avg_err = 1./(m.sqrt(np.sum(1./np.square(AR_errs))))

    # k is calculated as the low freq plateau value of alpha / AR
    k = alpha / AR_avg

    # And k_err is a sum of two relative errors * the absolute value of k
    k_err = (AR_avg_err/AR_avg + alpha_err/alpha) * k

    return k, k_err


def fitting_of_parameter_gamma_or_d(system_responses, mass, k, alpha):

    def AR_func(nu, c):
        return alpha/(np.sqrt((c * 2*np.pi*nu)**2 + (k - mass * 4.*np.pi*np.pi*nu*nu)**2))

    def phase_func(nu, c):
        return np.arctan2(-2*np.pi*nu * c, k - mass * (2*np.pi*nu)**2)

    def fit_func(nu, c):
        return np.concatenate([AR_func(nu, c), phase_func(nu, c)])

    ARs = np.array([sys_resp.AR for sys_resp in system_responses])
    phases = np.array([sys_resp.rod_phase for sys_resp in system_responses])
    freqs = np.array([sys_resp.rod_freq for sys_resp in system_responses])

    # BOUNDS MAY BE CONSTRICTING CORRECT c PARAMETER !
    coef, cov = curve_fit(fit_func, freqs, np.concatenate([ARs, phases]), p0=[0.0001], bounds=(0.00000001, 0.01), absolute_sigma=True)
    c = coef[0]
    print(c)
    c_err = m.sqrt(cov[0, 0])

    print("Final calibration fit!")
    fig, ax_AR, ax_phase = plot_sr(system_responses)
    ax_AR.plot(np.log10(freqs), np.log10(AR_func(freqs, c)))
    ax_phase.plot(np.log10(freqs), phase_func(freqs, c))
    plt.show()

    return c, c_err

# GLOBALS: - TODO: fix this so you dont have to define global variables!!!
low_border = -0.5
high_border = 0.5


def simple_calibration(**kwargs):
    kwargs["keyword_list"] = kwargs.get("keyword_list", []) + ["water"]
    system_responses = select_and_analyse(**kwargs)

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in system_responses])

    # Sort system responses by frequency
    system_responses = sorted(system_responses, key=lambda sys_resp: sys_resp.rod_freq)

    # Manually estimate freq range borders for low and high frequency regimes
    low_idx_border, high_idx_border = estimate_freq_range_borders(system_responses)

    # Calculate the current to force calibration constant alpha, from high freq system response
    alpha, alpha_err = calculate_force_current_factor(system_responses[high_idx_border:], rod.m, rod.m_err)

    # Calculate system compliance "k", for use in measurements.
    k, k_err = calculate_system_compliance(system_responses[:low_idx_border], alpha, alpha_err)

    # TODO: PREGLEJ ČE DELUJE VREDU OCENA NAPAKE c (TEGA ŠE NISI PREGLEDAL)
    c, c_err = fitting_of_parameter_gamma_or_d(system_responses, rod.m, k, alpha)

    cal_results = np.array([alpha, k, c, alpha_err, k_err, c_err])

    calibration = SimpleCalibration(cal_results, system_responses, rod, tub)
    ask_save_cal(calibration)

    return calibration


def FDM_calibration(**kwargs):
    """ STILL NOT FULLY CONVERGING!!! """

    kwargs["keyword_list"] = kwargs.get("keyword_list", []) + ["water"]
    system_responses = select_and_analyse(**kwargs)

    # Sort system responses by frequency
    system_responses = sorted(system_responses, key=lambda sys_resp: sys_resp.rod_freq)

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in system_responses])

    def construct_min_func(N):
        max_p = np.log(tub.W/rod.d)
        eta = 1.0034e-3     # in [Pa*s] @ 20C --> TODO: PREBERI IZ FILA
        rho = 998.2         # in [kg/m^3] @ 20C --> TODO: PREBERI IZ FILA

        omegas = np.array([resp.rod_freq*2*np.pi for resp in system_responses])
        ARs = np.array([resp.AR * np.exp(1.j * resp.rod_phase) for resp in system_responses])
        ARs_errors = np.array([(resp.AR_err/resp.AR + resp.rod_phase_err/resp.rod_phase)*ARs[i] for i, resp in enumerate(system_responses)])

        flowfields = np.zeros(shape=(len(omegas), N+2, N+2), dtype=np.complex128)
        for i, omega in enumerate(omegas):
            Re = rho * omega * ((rod.d/2.)**2) / eta
            g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, 0.0, Re)
            flowfields[i] = g
            # ps, thetas, hp in htheta pa bodo vedno enaki, zato bodo tudi po zadnji iteraciji vredu za uporabo naprej v funkciji

        def min_func(min_param):
            alpha, k = min_param[0], min_param[1]

            total_error = 0.
            for i, omega in enumerate(omegas):
                calculated_AR = alpha/(D_sub(flowfields[i], omega, eta, hp, htheta, rod.L) + k - rod.m*omega*omega)
                AR_calc_error = (calculated_AR - ARs[i])
                # RELATIVE ERROR, biased with measurements error
                # TODO BUILD BETTER FITTING ERROR FUNCTION, MAYBE BASED ON FITTING PHASE AND AMPLITUDE SEPARATELY
                # UTEZENO NAPAKA NA FAZI BOLI 2X toliko
                total_error += np.square(np.abs(np.real(AR_calc_error/ARs_errors[i]))) + 2*np.square(np.abs(np.imag(AR_calc_error/ARs_errors[i])))
            print("Trying with", alpha, k, "error:", total_error)
            return total_error
        
        return min_func 

    min_func = construct_min_func(30)

    # IZGLEDA DA JE BOLJE ČE STA ZAČETNA GUESSA PREMAJHNA!! (1e-7 << 1e-5 okoli kjer pričakujemo za rod_1003)
    min_res = minimize(min_func, np.array([1e-7, 1e-7]))
    calibration = FDMCalibration(min_res.x, system_responses, rod, tub)
    ask_save_cal(calibration)

    # PLOTTING ------------------------------------------------
    N = 30
    max_p = np.log(tub.W/rod.d)
    eta = 1.0034e-3     # in [Pa*s] @ 20C --> TODO: PREBERI IZ FILA
    rho = 998.2         # in [kg/m^3] @ 20C --> TODO: PREBERI IZ FILA

    omegas = np.array([resp.rod_freq*2*np.pi for resp in system_responses])
    ARs = np.array([resp.AR * np.exp(1.j * resp.rod_phase) for resp in system_responses])

    flowfields = np.zeros(shape=(len(omegas), N+2, N+2), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        Re = rho * omega * ((rod.d/2.)**2) / eta
        g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, 0.0, Re)
        flowfields[i] = g
    
    alpha, k = min_res.x[0], min_res.x[1]
    fig, ax_AR, ax_phase = plot_sr(system_responses)
    def complex_AR(omegas):
        res = []
        for i, omega in enumerate(omegas):
            res.append(alpha/(D_sub(flowfields[i], omega, eta, hp, htheta, rod.L) + k - rod.m*omega*omega))
        return res
    ax_AR.plot(np.log10(omegas/(2*np.pi)), np.log10(np.abs(complex_AR(omegas))))
    ax_phase.plot(np.log10(omegas/(2*np.pi)), np.angle(complex_AR(omegas)))
    plt.show()
    # PLOTTING ------------------------------------------------
    
    return calibration
    

def ask_save_cal(calibration):
    ans = str(input("Do you want to save the calibration? [Y/n] "))
    if ans == "Y":
        save_calibration(calibration)

