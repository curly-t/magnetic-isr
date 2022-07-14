import matplotlib.pyplot as plt
import numpy as np


def visualize_flowfield_polar(flowfield, ps, thetas, name=""):
    g = np.transpose(flowfield)

    print("Real")
    fig = plt.figure()
    rad = np.exp(ps)
    azm = thetas + 3 / 2 * np.pi
    r, th = np.meshgrid(rad, azm)
    z = np.real(g)

    plt.subplot(projection="polar", frameon=False)
    slika = plt.pcolormesh(th, r, z, shading="gouraud")
    plt.pcolormesh(th - np.pi / 2, r, z[::-1], shading="gouraud")  # ZRCALNA SLIKA
    plt.colorbar(slika)
    plt.tick_params(labelcolor="white", width=0)
    plt.tight_layout()
    if name != "":
        plt.savefig(name + "_real.pdf")
    plt.show()


    print("Imag")
    fig = plt.figure()
    rad = np.exp(ps)
    azm = thetas + 3 / 2 * np.pi
    r, th = np.meshgrid(rad, azm)
    z = np.imag(g)

    plt.subplot(projection="polar", frameon=False)
    slika = plt.pcolormesh(th, r, z, shading="gouraud")
    plt.pcolormesh(th - np.pi / 2, r, z[::-1], shading="gouraud")  # ZRCALNA SLIKA
    plt.colorbar(slika)
    plt.tick_params(labelcolor="white", width=0)
    plt.tight_layout()
    if name != "":
        plt.savefig(name + "_imag.pdf")
    plt.show()


def visualize_flowfield_rect(flowfield):
    g = np.transpose(flowfield)

    plt.imshow(np.real(g))
    plt.xlabel("p = log(r/a) [0, log(R/a)]")
    plt.ylabel("thetas [0, pi/2]")
    plt.show()


def plot_at_constant_theta(theta, flowfield, ps, thetas):
    theta_idx = np.argmin(np.abs(theta - thetas))

    rs = np.exp(ps)
    gs = flowfield[:, theta_idx]

    plt.plot(rs, np.real(gs), 'k-')
    plt.plot(rs, np.imag(gs), 'b--')
    plt.xlabel(r"$r/a$", fontsize=14)
    plt.ylabel(r"$g$    [arb.]", fontsize=14)
    plt.show()
