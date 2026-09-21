import numpy as np
from scipy.optimize import brentq
from turing.kinetics import steady_state, jacobian


def dispersion_matrix(k, a, b, gamma, D):
    """
    J - k² diag(1, D): linearised dynamics of a mode with wavenumber k.
    """
    J = jacobian(a, b, gamma)
    M = J - ((k**2) * np.diag([1, D]))

    return M


def growth_rate(k, a, b, gamma, D):
    """
    Largest real part of the eigenvalues of the dispersion matrix.
    """
    M = dispersion_matrix(k, a, b, gamma, D)
    eig = np.linalg.eigvals(M)

    return np.max(eig.real)


def growth_rate_analytical(k, a, b, gamma, D):
    """
    Same quantity (from growth_rate) via the trace/determinant formula.
    """
    M = dispersion_matrix(k, a, b, gamma, D)
    tr_M = np.trace(M)
    det_M = np.linalg.det(M)
    disc =  (tr_M**2) - (4 * det_M)

    return np.real((tr_M + np.emath.sqrt(disc))/2)


def det_coefficients(a, b, gamma, D):
    """
    (A, B, C) such that det(M) = A z² + B z + C, with z = k² (spatial wave number).
    """
    u, v = steady_state(a, b)

    A = D
    B = gamma * ((u**2) + D - (2 * D * u * v))
    C = (gamma**2) * (u**2)

    return A, B, C


def turing_discriminant(D, a, b, gamma):
    """
    B² - 4AC as a function of D. Its upper root is the threshold.
    """
    A, B, C = det_coefficients(a, b, gamma, D)

    return ((B**2) - (4*A*C))


def critical_D(a, b, gamma):
    """
    Critical diffusion ratio D_c, or nan if no Turing instability exists.
    """
    u, v = steady_state(a, b)
    
    if ((2 * u * v)-1) <= 0:
        return np.nan
    
    D_lo = (u**2) / ((2 * u * v) - 1)
    D_hi = 2 * D_lo
    while turing_discriminant(D_hi, a=a, b=b, gamma=gamma) <= 0:
        D_hi *= 2
    if np.trace(jacobian(a, b, gamma)) >= 0:
        return np.nan
    
    f_lo = turing_discriminant(D_lo, a, b, gamma)
    f_hi = turing_discriminant(D_hi, a, b, gamma)
    if f_hi * f_lo > 0:
        return np.nan
        
    D_c = brentq(turing_discriminant, D_lo, D_hi, args=(a, b, gamma))
    
    _, B_c, _ = det_coefficients(a, b, gamma, D_c)
    if B_c >= 0:
        return np.nan
    
    return D_c