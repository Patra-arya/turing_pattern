import numpy as np
import pytest
from turing.kinetics import jacobian
from turing.analysis import (
    dispersion_matrix, growth_rate, growth_rate_analytical,
    det_coefficients, critical_D
    )

# Reference parameters used throughout: the set in the notebook
# D_c here is known to four decimals.
a_ref, b_ref, gamma_ref, D_ref = 0.1, 0.9, 1000, 10


def test_dispersion_matrix_at_zero_k_equals_jacobian():
    """
    At k = 0 the diffusion term vanishes, so M must equal J exactly.
    """
    M = dispersion_matrix(0, a_ref, b_ref, gamma_ref, D_ref)
    J = jacobian(a_ref, b_ref, gamma_ref)

    np.testing.assert_allclose(M, J)


@pytest.mark.parametrize('k', [0.0, 5.0, 15.0, 18.7, 25.0, 50.0])
def test_growth_rate_numerical_matches_analytical(k):
    """
    Eigenvalue function and trace/determinant function give the same λ(k).
    """
    num = growth_rate(k, a_ref, b_ref, gamma_ref, D_ref)
    ana = growth_rate_analytical(k, a_ref, b_ref, gamma_ref, D_ref)

    np.testing.assert_allclose(num, ana, rtol = 1e-8)


def test_critical_D_matches_known_value():
    """
    D_c at (0.1, 0.9) matches the root of 0.64 D² − 5.6 D + 1 = 0 (hand calculation).
    """
    D_c = critical_D(a_ref, b_ref, gamma_ref)
    expected = (5.6 + np.sqrt((5.6**2) - (4 * 0.64 * 1))) / (2 * 0.64)

    assert np.isclose(D_c, expected, rtol = 1e-8)


def test_critical_D_is_nan_when_trace_positive():
    """
    No Turing instability when the uniform state is already unstable.
    """
    D_c = critical_D(0.1, 0.7, gamma_ref)

    assert np.isnan(D_c)


def test_critical_D_is_nan_without_self_activation():
    """
    No Turing instability when b < a, since ∂f/∂u ≤ 0 there.
    """
    D_c = critical_D(0.5, 0.1, gamma_ref)

    assert np.isnan(D_c)


@pytest.mark.parametrize("a, b", [(0.1, 0.9), (0.12, 1.0), (0.13, 2.25)])
def test_critical_D_independent_of_gamma(a, b):
    D1 = critical_D(a, b, 10)
    D2 = critical_D(a, b, 1e3)
    D3 = critical_D(a, b, 1e6)

    assert np.isclose(D1, D2)
    assert np.isclose(D1, D3) 
    assert np.isclose(D2, D3) 


def test_growth_rate_changes_sign_at_threshold():
    """
    Just below D_c every mode decays; just above, one grows.
    """
    D_c = critical_D(a_ref, b_ref, gamma_ref)
    assert not np.isnan(D_c)

    D_below = D_c * 0.99
    D_above = D_c * 1.01

    A_lo, B_lo, _ = det_coefficients(a_ref, b_ref, gamma_ref, D_below)
    z_lo = -B_lo / (2 * A_lo)
    k_lo = np.sqrt(z_lo)
    assert growth_rate(k_lo, a_ref, b_ref, gamma_ref, D_below) < 0

    A_hi, B_hi, _ = det_coefficients(a_ref, b_ref, gamma_ref, D_above)
    z_hi = -B_hi / (2 * A_hi)
    k_hi = np.sqrt(z_hi)
    assert growth_rate(k_hi, a_ref, b_ref, gamma_ref, D_above) > 0