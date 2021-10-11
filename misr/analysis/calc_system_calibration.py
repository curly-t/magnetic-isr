import math as m
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from scipy.optimize import curve_fit

from .freq_and_phase_extract import freq_phase_ampl
from .import_trackdata import select_filter_import_data


# MOGOČE TOLE NE SODI RAVNO SEM, ODLOČI SE KASNJE!
def plot_system_responses(system_responses, name=''):
    fig, (ax_AR, ax_phase) = plt.subplots(figsize=(8, 10), ncols=1, nrows=2, sharex=True)
    for i in range(len(system_responses)):

        freq = system_responses[i].rod_freq
        freq_err = system_responses[i].rod_freq_err
        AR = system_responses[i].AR
        AR_err = system_responses[i].AR_err
        phase = system_responses[i].rod_phase
        phase_err = system_responses[i].rod_phase_err

        ax_AR.errorbar(np.log10(freq), np.log10(AR), xerr=freq_err/freq, yerr=AR_err/AR,
                       elinewidth=1, capthick=1, capsize=2, markersize=3, marker='o', color='k')
        ax_phase.errorbar(np.log10(freq), phase, xerr=freq_err/freq, yerr=phase_err,
                          elinewidth=1, capthick=1, capsize=2, markersize=3, marker='o', color='k')

    ax_AR.set_ylabel(r'$\log_{10}(AR)$   (AR [m/A])', fontsize=14)
    ax_AR.set_xlim(-2, 2)
    ax_AR.set_ylim(-4, -0.5)
    ax_AR.grid()

    ax_phase.set_xlabel(r'$\log_{10}(\nu)$', fontsize=14)
    ax_phase.set_ylabel(r'$Phase$', fontsize=14)
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-np.pi, 0)
    ax_phase.grid()

    plt.tight_layout()

    if name != '':
        fig.savefig("System_response_for_calibration_{0}.png".format(name), dpi=300)
    
    return fig, ax_AR, ax_phase


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

    fig, ax_AR, ax_phase = plot_system_responses(system_responses)

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
        if m.log10(system_responses[i+1].rod_freq) >= low_border and m.log10(system_responses[i].rod_freq) < low_border:
            low_idx_border = i + 1
        elif m.log10(system_responses[i+1].rod_freq) >= high_border and m.log10(system_responses[i].rod_freq) < high_border:
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
    coef, cov = curve_fit(fit_func, freqs, np.concatenate([ARs, phases]), p0=[0.0001], bounds=(0.000001, 0.0001), absolute_sigma=True)
    c = coef[0]
    c_err = m.sqrt(cov[0, 0])

    print("Final calibration fit!")
    fig, ax_AR, ax_phase = plot_system_responses(system_responses)
    ax_AR.plot(np.log10(freqs), np.log10(AR_func(freqs, c)))
    ax_phase.plot(np.log10(freqs), phase_func(freqs, c))
    plt.show()

    return c, c_err


# System calibration coefficients are determined here.
# This program would likely return two calibration dicts
# One for the system properties, and one for the rod properties

# GLOBALS: - TODO: fix this so you dont have to define global variables!!!
low_border = -0.5
high_border = 0.5


def calculate_system_calibration(initialdir=None, Iampl="*", Ioffs="*", Ifreq="*", keyword_list=["water"], mass=0.0001, mass_err=0.00001, freq_err=0.001, complex_drift=True, plot_sys_resp=False, plot_track_results=False):
    measurements = select_filter_import_data(initialdir, Iampl, Ioffs, Ifreq, keyword_list)
    system_responses = []
    for measrmnt in measurements:
        system_responses.append(freq_phase_ampl(measrmnt, freq_err, plot_track_results=plot_track_results, complex_drift=complex_drift))

    # Sort system responses by frequency
    system_responses = sorted(system_responses, key=lambda sys_resp: sys_resp.rod_freq)

    # Manually estimate freq range borders for low and high frequency regimes
    low_idx_border, high_idx_border = estimate_freq_range_borders(system_responses)

    # Calculate the current to force calibration constant alpha, from high freq system response
    alpha, alpha_err = calculate_force_current_factor(system_responses[high_idx_border:], mass, mass_err)

    # Calculate system compliance "k", for use in measurements.
    k, k_err = calculate_system_compliance(system_responses[:low_idx_border], alpha, alpha_err)

    # PREGLEJ ČE DELUJE VREDU OCENA NAPAKE c
    c, c_err = fitting_of_parameter_gamma_or_d(system_responses, mass, k, alpha)

    print("DONE")

    return alpha, k, c, alpha_err, k_err, c_err

