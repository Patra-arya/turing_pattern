import numpy as np
import pytest
from turing.kinetics import steady_state
from turing.solver import laplacian, stable_dt, initial_condition, simulate


shape_ref, a_ref, b_ref, dx_ref, D_ref, ndim_ref, gamma_ref = (20, 20), 0.1, 0.9, 1e-2, 10, 2, 1000
n_steps, save_every = 1e4, 500

def test_laplacian_of_constant_is_zero():
    """
    A uniform field has zero curvature everywhere, including edges and corners.
    """
    uniform = np.ones((10, 10)) * 3
    L = laplacian(uniform, dx_ref)

    assert np.allclose(L, 0, atol=1e-8)


def test_laplacian_matches_second_derivative_in_interior():
    """
    For f(x) = x², the interior Laplacian equals 2.
    """
    x = np.arange(1, 11, dx_ref)
    field = x**2
    L = laplacian(field, dx_ref)
    int_L = L[1:-1]

    np.testing.assert_allclose(int_L, 2.0, rtol=1e-5)


def test_laplacian_conserves_total_mass():
    """
    With zero-flux boundaries, the Laplacian sums to zero over the grid.
    """
    x = np.arange(1, 11, dx_ref)
    L = laplacian(x, dx_ref)

    np.testing.assert_allclose(np.sum(L), 0, atol=1e-8)


def test_stable_dt_is_stricter_in_2d():
    """
    For the same dx and D, the 2D timestep is smaller than the 1D one.
    """

    assert stable_dt(dx_ref, D_ref, ndim=1) > stable_dt(dx_ref, D_ref, ndim=2)


def test_initial_condition_is_centred_on_steady_state():
    """
    Mean of the initial field is close to (u*, v*) and noise stays within amplitude.
    """
    amp = 0.01
    u0, v0 = initial_condition(shape_ref, a_ref, b_ref, amp=amp, seed=42)
    u_star, v_star = steady_state(a_ref, b_ref)

    np.testing.assert_allclose(np.mean(u0), u_star, atol=1e-2)
    np.testing.assert_allclose(np.mean(v0), v_star, atol=1e-2)

    assert np.all(np.abs(u0 - u_star) <= amp)
    assert np.all(np.abs(v0 - v_star) <= amp)



def test_steady_state_is_preserved_without_noise():
    """
    Starting exactly at (u*, v*) with no perturbation, the field does not move.
    """
    u0, v0 = initial_condition(shape_ref, a_ref, b_ref, amp=0)
    u_star, v_star = steady_state(a_ref, b_ref)

    u_hist, v_hist = simulate(
        u0, v0, a_ref, b_ref, gamma_ref, 
        dt= stable_dt(dx_ref, D_ref, 2), 
        n_steps=n_steps, save_every=save_every, D=D_ref, dx=dx_ref
    )

    np.testing.assert_allclose(u_hist[-1], u_star, atol=1e-12)
    np.testing.assert_allclose(v_hist[-1], v_star, atol=1e-12)



def test_simulate_output_shape():
    """
    History has one snapshot per save interval, plus the initial condition.
    """
    u0, v0 = initial_condition(shape_ref, a_ref, b_ref)
    dt = stable_dt(dx_ref, D_ref, ndim_ref)
    u_f, v_f = simulate(
        u0, v0, a_ref, b_ref, gamma_ref, 
        dt=dt, n_steps=n_steps, save_every=save_every, D=D_ref, dx=dx_ref
        )

    assert u_f.shape == ((n_steps/save_every + 1), *shape_ref)