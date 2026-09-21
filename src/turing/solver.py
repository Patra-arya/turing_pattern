import numpy as np
from turing.kinetics import steady_state, reaction


def laplacian(field, dx):
    """
    Discrete Laplacian of a 1D or 2D field with zero-flux boundaries.
    Returns an array of the same shape as `field`.
    """
    if field.ndim == 2:
        x_pad = np.pad(field, pad_width=1, mode="edge")
        
        center = x_pad[1:-1, 1:-1]
        up = x_pad[0:-2, 1:-1]
        down = x_pad[2:, 1:-1]
        left = x_pad[1:-1, 0:-2]
        right = x_pad[1:-1, 2:]
        
        d2 = ((up + down + left + right) - (4 * center)) / (dx**2)
        return d2

    elif field.ndim == 1:
        x_pad = np.pad(field, pad_width=1, mode='edge')

        left = x_pad[0:-2]
        center = x_pad[1:-1]
        right = x_pad[2:]

        d2 = ((left + right) - (2 * center)) / (dx**2)
        return d2

    else:
         return np.nan


def stable_dt(dx, D, ndim, safety=0.2):
    """
    Largest explicit-Euler timestep satisfying the diffusion stability bound,
    scaled by `safety`. `D` is the larger of the two diffusion coefficients.
    """
    max_D = max(1.0, D)
    return safety * (dx**2) / (2 * ndim * max_D)


def initial_condition(shape, a, b, amp=0.01, seed=0):
    """
    Uniform steady state (u*, v*) plus independent uniform noise in
    [-amplitude, amplitude] i.e.,'amp' at every grid point.

    Returns (u0, v0), each of the given shape.
    """
    rng = np.random.default_rng(seed)

    u, v = steady_state(a, b)
    u0 = u + rng.uniform(-amp, amp, size=shape)
    v0 = v + rng.uniform(-amp, amp, size=shape)

    return u0, v0



def simulate(u0, v0, a, b, gamma, D, dx, dt, n_steps, save_every):
    """
    Integrate the Schnakenberg reaction-diffusion system with explicit Euler.

    Returns (u_history, v_history): arrays of shape (n_saved, *u0.shape),
    where the first snapshot is the initial condition.
    """
    u = u0.copy()
    v = v0.copy()

    u_f, v_f = [], []
    
    for step in range(int(n_steps)):
        if step % save_every == 0:
            u_f.append(u.copy())
            v_f.append(v.copy())
    
        L_u = laplacian(u, dx)
        L_v = laplacian(v, dx)
    
        f_u, g_v = reaction(u, v, a, b, gamma)
    
        u = u + dt * (L_u + f_u)
        v = v + dt * ((D * L_v) + g_v)

    if (int(n_steps)) % save_every == 0:
        u_f.append(u.copy())
        v_f.append(v.copy())

    return np.asarray(u_f), np.asarray(v_f)
    