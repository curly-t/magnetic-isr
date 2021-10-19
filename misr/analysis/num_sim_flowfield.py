import math as m
import numpy as np


import numpy as np
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs as eigs_nonSym
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def FDM_matrix(N, ps, hp, htheta, Re, Bo):
    """Constructs the N x N matrix M for solving the problem:
            M g - i Re e^2p g = c
       which represents the solution to the Navier Stokes eq (with BC).
       FDM ... Finite Difference Method """
    diag_ps = np.ones((N+2)**2) * -2. / (hp*hp)
    diag_thetas = np.ones((N+2)**2) * -2. / (htheta * htheta)
    diag_p1 = np.ones((N+2)**2 - 1) / (htheta * htheta)
    diag_pN = np.ones((N+2)**2 - (N+2)) / (hp * hp)
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
    M[bc2_1] = -(2./(htheta**2) + 2./(hp**2) + 4.*Bo*np.exp(-ps[1:-1])/(2. * htheta*hp*hp))
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
    
    return csr_matrix(M - B), c


def solve_Eigenvalue_problem_numpy(N):
    vals, vects = eigh(FDM_matrix(N))
    eig_vects = []
    for i in range(len(vects[0])):
        eig_vects.append(vects[:, i])
    return vals, eig_vects

def solve_Eigenvalue_problem_sparse(N):
    vals, vects = eigs_nonSym(FDM_matrix(N), k=1, which='SM')
    eig_vects = []
    for i in range(len(vects[0])):
        eig_vects.append(vects[:, i])
    return np.abs(vals), np.abs(eig_vects)
