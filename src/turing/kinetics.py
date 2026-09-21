import numpy as np

def steady_state(a, b):
    """
    Homogeneous steady state (u*, v*) of Schnakenberg kinetics.
    """
    u_star = a + b
    v_star = b / ((a+b)**2)
    return u_star, v_star

def reaction(u, v, a, b, gamma):
    """
    Reaction terms (f, g). Works on scalars or arrays.
    """
    f = gamma * (a - u + (u**2 * v))
    g = gamma * (b - (u**2 * v))
    return f, g

def jacobian(a, b, gamma):
    """
    2x2 Jacobian of the kinetics, evaluated at the steady state.
    """
    u, v = steady_state(a, b)

    j11 = (2 * gamma * u * v) - gamma
    j12 = gamma * (u**2)
    j21 = -(2 * gamma * u * v)
    j22 = -(gamma * (u**2))

    return np.array([[j11, j12],
                    [j21, j22]])