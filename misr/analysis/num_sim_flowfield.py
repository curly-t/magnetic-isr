import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import simpson

def construct_FDM_system(N, ps, hp, htheta, Re, Bo):
    """Constructs the N x N matrix M* for solving the problem:
            M g - i Re e^2p g = c
            M* g = c
       which represents the solution to the Navier Stokes eq (with BC).
       FDM ... Finite Difference Method
       N x N internal points."""
    diag_ps = np.ones((N+2)**2, dtype=np.complex128) * -2. / (hp*hp)
    diag_thetas = np.ones((N+2)**2, dtype=np.complex128) * -2. / (htheta * htheta)
    diag_p1 = np.ones((N+2)**2 - 1, dtype=np.complex128) / (htheta * htheta)
    diag_pN = np.ones((N+2)**2 - (N+2), dtype=np.complex128) / (hp * hp)
    M = np.diag(diag_ps) + np.diag(diag_thetas) + np.diag(diag_p1, k=1) + np.diag(diag_p1, k=-1) + np.diag(diag_pN, k=N+2) + np.diag(diag_pN, k=-(N+2))

    # Začetni blok identitet za robni pogoj pri i = 0 (p=0)
    M[:N+2, :N+2] = np.identity(N+2)
    M[:N+2, N+2:] = 0.
    
    # Končni blok identitet za robni pogoj pri i = N+1 (p=ln(R/a))
    M[-(N+2):, -(N+2):] = np.identity(N+2)
    M[-(N+2):, :-(N+2)] = 0.

    # Popravi nekaj ničel
    M[N+2::N+2, N+1::N+2] = 0.
    M[N+1::N+2, N+2::N+2] = 0.

    # Drugi odvod (O(h^2)) pri orbu j = 0 (theta = 0) (tu je 1. odvod 0 zato d^2g/dx^2 = (2g(x+h) - 2g(x))/(h**2)   ... ker g(x+h) = g(x-h) ker 1. odvod je 0)
    bc1_2 = (np.arange(N+2, (N+2)**2 - (N+2), N+2), np.arange(N+3, (N+2)**2 - (N+2), N+2))
    M[bc1_2] = 2./(htheta**2)

    # Drugi odvod (O(h^2)) pri orbu j = N+1 (theta = pi/2)  (bolj zahtevni robni pogoji)
    bc2_1 = (np.arange(2*N+3, (N+2)**2 - (N+2), N+2), np.arange(2*N+3, (N+2)**2 - (N+2), N+2))
    bc2_2 = (np.arange(2*N+3, (N+2)**2 - (N+2), N+2), np.arange(2*N+2, (N+2)**2 - (N+2), N+2))
    bc2_3 = (np.arange(2*N+3, (N+2)**2 - (N+2), N+2), np.arange(2*N+3 + N+2, (N+2)**2, N+2))
    bc2_4 = (np.arange(2*N+3, (N+2)**2 - (N+2), N+2), np.arange(2*N+3 - (N+2), (N+2)**2 - 2*(N+2), N+2))
    M[bc2_1] = -(2./(htheta**2) + 2./(hp**2) + 4.*Bo*np.exp(-ps[1:-1])/(htheta*hp*hp))
    M[bc2_2] = 2. / (htheta**2)
    M[bc2_3] = 1./(hp**2) + 2.*Bo*np.exp(-ps[1:-1])/(hp*hp*htheta) - Bo*np.exp(-ps[1:-1])/(hp*htheta)
    M[bc2_4] = 1./(hp**2) + 2.*Bo*np.exp(-ps[1:-1])/(hp*hp*htheta) + Bo*np.exp(-ps[1:-1])/(hp*htheta)

    # Second matrix B
    B_in = np.identity((N+2)**2 - 2*(N+2)) * 1.j * np.exp(2 * np.repeat(ps[1:-1], N+2)) * Re
    B = np.zeros(((N+2)**2, (N+2)**2), dtype=np.complex128)
    B[N+2:-(N+2), N+2:-(N+2)] = B_in

    # vektor za določanje RP
    c = np.zeros((N+2)**2)
    c[:N+2] = 1.0
    
    # Mogoče deluje pravilno? Nevem čist točno?
    return csr_matrix(M - B), c


def flowfield_FDM(N, max_p, Bo, Re):
    """Calculate the flowfield with the FDM.
        Returns a 2D array in the space (p, theta).
        The first index runs along the p axis from 0 to max_p (both limits included),
        The second index rund along the theta axis from 0 to pi/2 (both limits included)."""
        
    ps = np.linspace(0, max_p, num=N+2)
    thetas = np.linspace(0, np.pi/2, num=N+2)

    hp = ps[1] - ps[0]
    htheta = thetas[1] - thetas[0]

    M, c = construct_FDM_system(N, ps, hp, htheta, Re, Bo)

    return np.reshape(spsolve(M, c), (N+2, N+2)), ps, thetas, hp, htheta


def dgdp_at_p_0(g, hp):
    return (-3.*g[0] + 4.*g[1] - g[2])/(2. * hp)


def D_sub(g, omega, eta, hp, htheta, L):
    integral = simpson(-dgdp_at_p_0(g, hp), dx=htheta)
    # Returns separately the (real, imaginary) pair of arrays.
    return np.array([-2*L*omega*eta*np.imag(integral), 2*L*omega*eta*np.real(integral)])


def D_surf(g, cplxBo, omega, eta, hp, L):
    dgdp_Re, dgdp_Im = np.real(dgdp_at_p_0(g[:, -1], hp)), np.imag(dgdp_at_p_0(g[:, -1], hp))
    # Returns separately the (real, imaginary) pair of arrays.
    return np.array([2*L*omega*eta*(cplxBo[0]*dgdp_Im + dgdp_Re*cplxBo[1]),
                     2*L*omega*eta*(cplxBo[1]*dgdp_Im - cplxBo[0]*dgdp_Re)])

