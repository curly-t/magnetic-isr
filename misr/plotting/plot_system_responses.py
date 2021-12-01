import matplotlib.pyplot as plt
import numpy as np
from gvar import sdev
from gvar import mean as gvalue
from ..analysis.num_sim_flowfield import flowfield_FDM, D_sub


def plot_sr(system_responses, name='', color='k', marker='o', fig=None, ax_AR=None, ax_phase=None):
    if (fig is None) or (ax_AR is None) or (ax_phase is None):
        fig, (ax_AR, ax_phase) = plt.subplots(figsize=(8, 10), ncols=1, nrows=2, sharex=True)

    for i in range(len(system_responses)):
        log10freq = np.log(system_responses[i].rod_freq)/np.log(10)
        log10AR = np.log(system_responses[i].AR)/np.log(10)
        phase = system_responses[i].rod_phase

        ax_AR.errorbar(gvalue(log10freq), gvalue(log10AR), xerr=sdev(log10freq), yerr=sdev(log10AR),
                       elinewidth=1, capthick=1, capsize=2, markersize=3, marker=marker, color=color)
        ax_phase.errorbar(gvalue(log10freq), gvalue(phase), xerr=sdev(log10freq), yerr=sdev(phase),
                          elinewidth=1, capthick=1, capsize=2, markersize=3, marker=marker, color=color)

    ax_AR.set_ylabel(r'$\log_{10}(AR)$   (AR [m/A])', fontsize=14)
    ax_AR.grid()

    ax_phase.set_xlabel(r'$\log_{10}(\nu)$', fontsize=14)
    ax_phase.set_ylabel(r'$Phase$', fontsize=14)
    ax_phase.grid()

    plt.tight_layout()

    if name != '':
        fig.savefig("System_response_{0}.png".format(name), dpi=300)

    return fig, ax_AR, ax_phase


def plot_fit_results_pointwise(fig, ax_AR, ax_phase, ARs, phases, freqs):
    ax_AR.plot(np.log10(gvalue(freqs)), gvalue(np.log(ARs)/np.log(10)))
    ax_AR.fill_between(np.log10(gvalue(freqs)),
                       gvalue(np.log(ARs)/np.log(10)) - sdev(np.log(ARs)/np.log(10)),
                       gvalue(np.log(ARs)/np.log(10)) + sdev(np.log(ARs)/np.log(10)),
                       alpha=0.3)

    ax_phase.plot(np.log10(gvalue(freqs)), gvalue(phases))
    ax_phase.fill_between(np.log10(gvalue(freqs)),
                          gvalue(phases) - sdev(phases),
                          gvalue(phases) + sdev(phases),
                          alpha=0.3)

    return fig, ax_AR, ax_phase


def plot_FDM_cal_fit(fig, ax_AR, ax_phase, alpha, k, freqs, rod, tub, N=30, eta=1.0034e-3, rho=998.2):
    max_p = gvalue(np.log(tub.W / rod.d))
    omegas = gvalue(freqs*2*np.pi)

    flowfields = np.zeros(shape=(len(omegas), N + 2, N + 2), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        Re = rho * omega * ((gvalue(rod.d) / 2.) ** 2) / eta
        g, ps, thetas, hp, htheta = flowfield_FDM(N, max_p, 0.0, Re)
        flowfields[i] = g

    def calculated_ARs_and_phases(omegas):
        res = []
        for i, omega in enumerate(omegas):
            D_sub_real, D_sub_imag = D_sub(flowfields[i], omega, eta, hp, htheta, rod.L)
            D_re = D_sub_real + k - rod.m * omega * omega
            D_im = D_sub_imag
            calc_imag_AR = - alpha * D_im / (np.square(D_im) + np.square(D_re))
            calc_real_AR = alpha * D_re / (np.square(D_im) + np.square(D_re))

            calc_AR = np.sqrt(np.square(calc_real_AR) + np.square(calc_imag_AR))
            calc_phase = np.arctan2(calc_imag_AR, calc_real_AR)

            res.append(np.array([calc_AR, calc_phase]))
        res = np.array(res)
        return res[:, 0], res[:, 1]

    ARs, phases = calculated_ARs_and_phases(omegas)

    return plot_fit_results_pointwise(fig, ax_AR, ax_phase, ARs, phases, freqs)
