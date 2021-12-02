import matplotlib.pyplot as plt

from gvar import sdev
from gvar import mean as gvalue


def plot_cplx_modulus(final_results):
    if len(final_results.G) > 0:
        freqs = final_results.freqs
        G_Re = final_results.G[:, 0]
        G_Im = final_results.G[:, 1]

        plt.errorbar(gvalue(freqs), gvalue(G_Re), yerr=sdev(G_Re), xerr=sdev(freqs), color="g", linewidth=0,
                     elinewidth=1, capsize=1, capthick=1, marker="o", label="G'")
        plt.errorbar(gvalue(freqs), gvalue(G_Im), yerr=sdev(G_Im), xerr=sdev(freqs), color="b", linewidth=0,
                     elinewidth=1, capsize=1, capthick=1, marker="o", label="G''")

    if len(final_results.excl_G) > 0:
        excl_freqs = final_results.excl_freqs
        excl_G_Re = final_results.excl_G[:, 0]
        excl_G_Im = final_results.excl_G[:, 1]
        plt.errorbar(gvalue(excl_freqs), gvalue(excl_G_Re), yerr=sdev(excl_G_Re), xerr=sdev(excl_freqs), color="g",
                     linewidth=0, elinewidth=1, capsize=1, capthick=1, marker="o", alpha=0.3, label="G'  low Bo")
        plt.errorbar(gvalue(excl_freqs), gvalue(excl_G_Im), yerr=sdev(excl_G_Im), xerr=sdev(excl_freqs), color="b",
                     linewidth=0, elinewidth=1, capsize=1, capthick=1, marker="o", alpha=0.3, label="G'' low Bo")
    plt.legend()
    plt.xlabel(r"$\nu$", fontsize=14)
    plt.ylabel("G' , G''", fontsize=14)
    plt.title(r"Complex G as a function of frequency $\nu$", fontsize=14)
    plt.tight_layout()
    plt.show()

