import numpy as np
import pytest
from turing.kinetics import (
    steady_state, reaction, jacobian
    )

def test_steady_state_is_fixed_point():
    """
    Reaction terms vanish at (u*, v*).
    """
    a, b, gamma = 0.1, 0.9, 1000
    u, v = steady_state(a, b)
    react = reaction(u, v, a, b, gamma)

    np.testing.assert_allclose(react, 0, atol=1e-10)



def test_jacobian_trace_matches_hand_derivation():
    """
    tr(J) = -0.2𝛄 at a=0.1, b=0.9.
    """
    a, b, gamma = 0.1, 0.9, 1000
    J = jacobian(a, b, gamma)
    assert np.isclose(np.trace(J), (-0.2*gamma))


def test_jacobian_determinant_matches_hand_derivation():
    """
    det(J) = γ² at a=0.1, b=0.9.
    """
    a, b, gamma = 0.1, 0.9, 1000
    J = jacobian(a, b, gamma)

    assert np.isclose(np.linalg.det(J), (gamma**2))


def _finite_difference_jacobian(a, b, gamma, h=1e-6):
    """
    Estimate J at the steady state by nudging u and v in reaction().
    """
    u, v = steady_state(a, b)
    col_u = (np.array(reaction(u + h, v, a, b, gamma))
         - np.array(reaction(u - h, v, a, b, gamma))) / (2 * h)
    col_v = (np.array(reaction(u, v + h, a, b, gamma))
             - np.array(reaction(u, v - h, a, b, gamma))) / (2 * h)
    J_s = np.column_stack([col_u, col_v])

    return J_s


def test_jacobian_matches_finite_difference():
    """
    Analytical J agrees with a central-difference estimate from reaction().
    """
    a, b, gamma = 0.1, 0.9, 1000
    J = jacobian(a, b, gamma)
    J_s = _finite_difference_jacobian(a, b, gamma)

    np.testing.assert_allclose(J_s, J, rtol=1e-5)