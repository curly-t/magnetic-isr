import math as m
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit, leastsq
from gvar import mean as gvalue
from gvar import sdev, gvar

from .freq_and_phase_extract import select_and_analyse

from .num_sim_flowfield import flowfield_FDM, D_sub
from .Rod_TubClass import get_Rod_and_Tub

from .CalibrationClass import SimpleCalibration, FDMCalibration
from ..utils.save_and_get_calibration import save_calibration

from ..plotting.plot_system_responses import plot_sr, plot_fit_results_pointwise, plot_FDM_cal_fit


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
        if m.log10(gvalue(system_responses[i + 1].rod_freq)) >= low_border > m.log10(gvalue(system_responses[i].rod_freq)):
            low_idx_border = i + 1
        elif m.log10(gvalue(system_responses[i + 1].rod_freq)) >= high_border > m.log10(gvalue(system_responses[i].rod_freq)):
            high_idx_border = i + 1
    
    return low_idx_border, high_idx_border


def calculate_force_current_factor(high_freq_system_responses, mass, normal_power_tol=1):
    # Construct arrays for line fitting
    log10ARs = np.log([sys_resp.AR for sys_resp in high_freq_system_responses])/np.log(10)
    log10freqs = np.log([sys_resp.rod_freq for sys_resp in high_freq_system_responses])/np.log(10)

    # Fit a line trough the high frequency regime data points
    # Error is returned unscaled, wights must satisfy: weights = 1/sigma**2
    high_freq_AR_coefs, high_freq_AR_cov = np.polyfit(gvalue(log10freqs), gvalue(log10ARs),
                                                      deg=1,
                                                      w=1. / np.square(sdev(log10ARs)),
                                                      cov='unscaled')

    high_freq_AR_coefs = gvar(high_freq_AR_coefs, high_freq_AR_cov)

    if gvalue((high_freq_AR_coefs[0] + 2)/m.sqrt(high_freq_AR_cov[0, 0])) >= normal_power_tol:
        warnings.warn(f"Potencna odvisnost omege pri visokih frekvencah je {high_freq_AR_coefs[0]}!")
    
    # Intermediate constant, A = alpha/(4*m*pi**2)
    A = 10**(high_freq_AR_coefs[1])
    
    # Final current to force converting constant alpha, a property of the rod - alpha is in units [N/A]
    alpha = 4. * (np.pi ** 2) * mass * A

    return alpha


def calculate_system_compliance(low_freq_system_responses, alpha):
    ARs = [sys_resp.AR for sys_resp in low_freq_system_responses]

    # Optimal combination of ARs is to weight them by weights = 1./sigma**2 (Wikipedia)
    weights = 1./np.square(sdev(ARs))
    AR_avg = np.sum(weights * ARs)/np.sum(weights)

    # k is calculated as the low freq plateau value of alpha / AR
    k = alpha / AR_avg

    return k


def fitting_of_parameter_gamma_or_d(system_responses, mass, k, alpha):
    # Zavestno smo se odločili in tu upoštevali MEANS of k and mass, ker bi blo sicer težko vključit v curve_fit!
    # In itak je simple calibration le proof of concept, ni treba da je dejst natančno, uporabljalo bi se potem FDM star
    def AR_func(nu, c):
        return gvalue(alpha)/(np.sqrt((c * 2*np.pi*nu)**2 + (gvalue(k) - gvalue(mass) * 4.*np.pi*np.pi*nu*nu)**2))

    def phase_func(nu, c):
        return np.arctan2(-2*np.pi*nu * c, gvalue(k) - gvalue(mass) * (2*np.pi*nu)**2)

    def fit_func(nu, c):
        return np.concatenate([AR_func(nu, c), phase_func(nu, c)])

    ARs = np.array([sys_resp.AR for sys_resp in system_responses])
    phases = np.array([sys_resp.rod_phase for sys_resp in system_responses])
    freqs = np.array([sys_resp.rod_freq for sys_resp in system_responses])

    # BOUNDS MAY BE CONSTRICTING CORRECT c PARAMETER!
    coef, cov = curve_fit(fit_func, gvalue(freqs), gvalue(np.concatenate([ARs, phases])), p0=[0.0001],
                          bounds=(0.00000001, 0.01), absolute_sigma=True, sigma=sdev(np.concatenate([ARs, phases])))
    coef = gvar(coef, cov)
    c = coef[0]

    print("Final calibration fit!")
    ARs = AR_func(freqs, c)
    phases = phase_func(freqs, c)
    fig, ax_AR, ax_phase = plot_sr(system_responses)
    plot_fit_results_pointwise(fig, ax_AR, ax_phase, ARs, phases, freqs)
    plt.show()

    return c


# GLOBALS: - TODO: fix this so you don't have to define global variables!!!
low_border = -0.5
high_border = 0.5


def simple_calibration(**kwargs):
    kwargs["keyword_list"] = kwargs.get("keyword_list", []) + ["water"]
    system_responses = select_and_analyse(**kwargs)

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in system_responses])

    # Sort system responses by frequency
    system_responses = sorted(system_responses, key=lambda sys_resp: gvalue(sys_resp.rod_freq))

    # Manually estimate freq range borders for low and high frequency regimes
    low_idx_border, high_idx_border = estimate_freq_range_borders(system_responses)

    # Calculate the current to force calibration constant alpha, from high freq system response
    alpha = calculate_force_current_factor(system_responses[high_idx_border:], rod.m)

    # Calculate system compliance "k", for use in measurements.
    k = calculate_system_compliance(system_responses[:low_idx_border], alpha)

    c = fitting_of_parameter_gamma_or_d(system_responses, rod.m, k, alpha)

    cal_results = np.array([alpha, k, c])

    calibration = SimpleCalibration(cal_results, system_responses, rod, tub)
    ask_save_cal(calibration)

    return calibration


def FDM_calibration(**kwargs):
    """ NOT CONVERGING TILL THE GVAR IMPLEMENTATION! """

    kwargs["keyword_list"] = kwargs.get("keyword_list", []) + ["water"]
    system_responses = select_and_analyse(**kwargs)

    # Sort system responses by frequency
    system_responses = sorted(system_responses, key=lambda sys_resp: gvalue(sys_resp.rod_freq))

    rod, tub = get_Rod_and_Tub([sr.meas.dirname for sr in system_responses])

    def construct_min_func(N):
        max_p = gvalue(np.log(tub.W/rod.d))
        eta = 1.0034e-3     # in [Pa*s] @ 20C --> TODO: PREBERI IZ FILA
        rho = 998.2         # in [kg/m^3] @ 20C --> TODO: PREBERI IZ FILA

        # Beware - omegas are just floats, is easier.
        omegas = gvalue(np.array([resp.rod_freq*2*np.pi for resp in system_responses]))

        flowfields = np.zeros(shape=(len(omegas), N+2, N+2), dtype=np.complex128)
        for i, omega in enumerate(omegas):
            Re = rho * omega * ((gvalue(rod.d)/2.)**2) / eta    # Zaradi simulacije delamo z float in ne gvar samo tu.
            g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, 0.0, Re)
            flowfields[i] = g
            # ps, thetas, hp in htheta pa bodo vedno enaki, zato bodo tudi po zadnji iteraciji vredu za uporabo naprej v funkciji

        def min_func(min_param):
            alpha, k = min_param[0], min_param[1]

            errors = np.zeros(2*len(flowfields))
            for i, omega in enumerate(omegas):
                D_sub_real, D_sub_imag = D_sub(flowfields[i], omega, eta, hp, htheta, rod.L)

                D_re = D_sub_real + k - rod.m * omega * omega
                D_im = D_sub_imag
                calc_imag_AR = - alpha * D_im / (np.square(D_im) + np.square(D_re))
                calc_real_AR = alpha * D_re / (np.square(D_im) + np.square(D_re))

                calc_AR = np.sqrt(np.square(calc_real_AR) + np.square(calc_imag_AR))
                calc_phase = np.arctan2(calc_imag_AR, calc_real_AR)

                # Takle glupo morem delat ker gvar ne podpira np.abs()
                AR_calc_error = np.sqrt(np.square(calc_AR - system_responses[i].AR))
                phase_calc_error = np.sqrt(np.square(calc_phase - system_responses[i].rod_phase))

                # FOLLOWING CHANGE MAY ALSO CAUSE INSTABILITY; BUT SHOULD BE MORE TRUE!
                # errors[i] = AR_calc_error/sdev(system_responses[i].AR)
                # errors[len(omegas) + i] = phase_calc_error/sdev(system_responses[i].rod_phase)
                errors[i] = gvalue(AR_calc_error / sdev(AR_calc_error))
                errors[len(omegas) + i] = gvalue(phase_calc_error/sdev(phase_calc_error))
            return errors

        return min_func, omegas

    min_func, omegas = construct_min_func(30)

    # USES METHOD "lm"
    best_params, scaled_cov, info_dict, msg, ier = leastsq(min_func, np.array([1e-7, 1e-7]), ftol=1e-12, full_output=True)

    # As described in the scipy.optimize.leastsq docs.
    cov = scaled_cov * np.var(min_func(best_params)) / (len(omegas) - 2)
    best_params = gvar(best_params, cov)

    calibration = FDMCalibration(best_params, system_responses, rod, tub)

    # Plotting
    fig, ax_AR, ax_phase = plot_sr(system_responses)
    plot_FDM_cal_fit(fig, ax_AR, ax_phase, best_params[0], best_params[1], omegas/(2*np.pi), rod, tub)
    plt.show()

    ask_save_cal(calibration)

    return calibration
    

def ask_save_cal(calibration):
    ans = str(input("Do you want to save the calibration? [Y/n] "))
    if ans == "Y":
        save_calibration(calibration)

