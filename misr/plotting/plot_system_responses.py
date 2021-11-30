import matplotlib.pyplot as plt
import numpy as np
from gvar import sdev
from gvar import mean as gvalue


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
