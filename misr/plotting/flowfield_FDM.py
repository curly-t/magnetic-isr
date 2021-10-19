import matplotlib.pyplot as plt
import numpy as np


def visualize_flowfield(flowfield, ps, thetas):
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
