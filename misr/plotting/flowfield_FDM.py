import matplotlib.pyplot as plt
import numpy as np


def visualize_flowfield_polar(flowfield, ps, thetas):
    g = np.transpose(flowfield)

    fig = plt.figure()
    rad = np.exp(ps)
    azm = thetas + 3/2 * np.pi
    r, th = np.meshgrid(rad, azm)
    z = np.real(g)

    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, z, shading="gouraud")
    plt.pcolormesh(th - np.pi/2, r, z[::-1], shading="gouraud")     # ZRCALNA SLIKA
    plt.grid()
    plt.show()


def visualize_flowfield_rect(flowfield):
    g = np.transpose(flowfield)

    plt.imshow(np.real(g))
    plt.xlabel("p = log(r/a) [0, log(R/a)]")
    plt.ylabel("thetas [0, pi/2]")
    plt.show()
