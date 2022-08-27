from math import pi
from gvar import gvar


def calc_mass(OD_glass, ID_glass, D_Ni, L_glass, L_Ni, rho_glass=2.23, rho_Ni=8.908):
    """ Calculates the mass of the rod, based on measurements.
    Arguments may be input as gvar.Gvar or float type. All measurements of
    D are in mum, all measurements of L are in mm, mass is given in mg.
    Density is input as mg/mm**3 or g/cm**3. """

    # Inside all calc preformed using mm
    ro_glass = OD_glass / 2000
    ri_glass = ID_glass / 2000
    r_Ni = D_Ni / 2000
    return pi * ((ro_glass**2 - ri_glass**2) * L_glass * rho_glass + (r_Ni**2) * L_Ni * rho_Ni)


if __name__ == "__main__":
    # Samo za hitro racunanje mase
    OD_glass = gvar(400, 12)    # Proizvajalec garantira 10% napako,
    ID_glass = gvar(300, 8)     # jst predpostavljam 3% ker drugače pride napaka way to big
    D_Ni = gvar(250, 5)
    L_Ni = gvar(3.94, 0.02)
    L_glass = gvar(30.97, 0.02)

    print(f"{calc_mass(OD_glass, ID_glass, D_Ni, L_glass, L_Ni)} mg")

