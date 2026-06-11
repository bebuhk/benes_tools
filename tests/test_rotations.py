"""Consistency tests for benes_tools.rotations.

These verify that the conversion functions in rotations.py form a consistent
round-trip:

    super_fibonacci(n)      ->  unit quaternions          (n, 4)
    quats_to_matrices       ->  rotation matrices         (n, 3, 3)
    matrix_to_euler_zyz     ->  z-y-z Euler angles        (n, 3)
    euler_zyz_to_quat       ->  unit quaternions again    (n, 4)

If every function agrees on conventions, the final quaternions must describe
the same rotations as the originals.

Note on the comparison: unit quaternions are a *double* cover of SO(3), so q
and -q represent the identical rotation. We therefore compare orientations,
not raw components -- either via |<q1, q2>| ~ 1 or by comparing the rotation
matrices the quaternions produce.
"""

# x64 must be enabled before the first JAX computation, otherwise we'd be
# testing the math through float32 round-off rather than the math itself.
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest


# Below some special import from claude that avoids specifying all packages from benes_tools in the pyproject.
# # Import rotations.py in isolation. The benes_tools package __init__ eagerly
# # imports sibling modules with undeclared dependencies (e.g. si_units), which
# # is unrelated to the rotation math under test -- so we load the module file
# # directly instead of `from benes_tools.rotations import ...`.
# import importlib.util
# from pathlib import Path
# _spec = importlib.util.spec_from_file_location(
#     "benes_tools_rotations",
#     Path(__file__).resolve().parent.parent / "benes_tools" / "rotations.py",
# )
# rotations = importlib.util.module_from_spec(_spec)
# _spec.loader.exec_module(rotations)
# super_fibonacci = rotations.super_fibonacci
# quats_to_matrices = rotations.quats_to_matrices
# matrix_to_euler_zyz = rotations.matrix_to_euler_zyz
# euler_zyz_to_quat = rotations.euler_zyz_to_quat

# for now i opted to import all dependencies in pyproject.toml, which allows the standard import here:
from benes_tools.rotations import super_fibonacci, quats_to_matrices, matrix_to_euler_zyz, euler_zyz_to_quat, euler_zyz


def _roundtrip_quats(n):
    """Run the full super_fibonacci -> ... -> euler_zyz_to_quat pipeline."""
    q = super_fibonacci(n, dtype=jnp.float64)
    R = quats_to_matrices(q)
    angles = matrix_to_euler_zyz(R)          # (n, 3): columns phi, theta, psi
    phi, theta, psi = angles[:, 0], angles[:, 1], angles[:, 2]
    q2 = jax.vmap(euler_zyz_to_quat)(phi, theta, psi)
    return q, q2


@pytest.mark.parametrize("n", [10, 100, 1000, 4096])
def test_quaternion_roundtrip_is_identity(n):
    """The quaternions surviving the round-trip match the originals (up to sign)."""
    q, q2 = _roundtrip_quats(n)

    # both ends must be unit quaternions
    assert jnp.allclose(jnp.linalg.norm(q, axis=-1), 1.0, atol=1e-9)
    assert jnp.allclose(jnp.linalg.norm(q2, axis=-1), 1.0, atol=1e-9)

    # |<q, q2>| == 1  <=>  q2 == +-q  <=>  same rotation
    dots = jnp.abs(jnp.sum(q * q2, axis=-1))
    max_err = float(jnp.max(jnp.abs(dots - 1.0)))
    assert max_err < 1e-9, f"max |<q,q2>| deviation from 1: {max_err:.2e}"


@pytest.mark.parametrize("n", [10, 100, 1000, 4096])
def test_roundtrip_matrices_match(n):
    """Independent check: the rotation matrices must agree exactly.

    This is sign-agnostic by construction (q and -q give the same matrix), so
    it cross-checks the |<q,q2>| test above without relying on it.
    """
    q, q2 = _roundtrip_quats(n)
    R1 = quats_to_matrices(q)
    R2 = quats_to_matrices(q2)
    max_err = float(jnp.max(jnp.abs(R1 - R2)))
    assert max_err < 1e-9, f"max rotation-matrix deviation: {max_err:.2e}"


@pytest.mark.parametrize("n", [10, 100, 1000, 4096])
def test_roundtrip_angles_match(n):
    """Independent check: the rotation angles must agree exactly.

    This is sign-agnostic by construction (q and -q give the same matrix), so
    it cross-checks the |<q,q2>| test above without relying on it.
    """
    q, q2 = _roundtrip_quats(n)
    R1 = quats_to_matrices(q)
    R2 = quats_to_matrices(q2)
    angles1 = matrix_to_euler_zyz(R1)
    angles2 = matrix_to_euler_zyz(R2)
    max_err = float(jnp.max(jnp.abs(angles1 - angles2)))
    assert max_err < 1e-9, f"max rotation-angles deviation: {max_err:.2e}"

    phi1, theta1, psi1 = angles1[:, 0], angles1[:, 1], angles1[:, 2]
    #print(f"max angles1: {jnp.max(angles1)}, min angles1: {jnp.min(angles1)}, mean angles1: {jnp.mean(angles1)}")
    #R1_2 = jax.vmap(euler_zyz)(phi1, theta1, psi1)
    R1_2 = jax.vmap(lambda phi_, theta_, psi_: euler_zyz(phi_, theta_, psi_, degrees=False))(phi1, theta1, psi1)
    phi2, theta2, psi2 = angles2[:, 0], angles2[:, 1], angles2[:, 2]
    #print(f"max angles2: {jnp.max(angles2)}, min angles2: {jnp.min(angles2)}, mean angles2: {jnp.mean(angles2)}")
    #print(f"max phi1: {jnp.max(phi1)}, min phi1: {jnp.min(phi1)}, mean phi1: {jnp.mean(phi1)}")
    #print(f"max theta1: {jnp.max(theta1)}, min theta1: {jnp.min(theta1)}, mean theta1: {jnp.mean(theta1)}")
    #print(f"max psi1: {jnp.max(psi1)}, min psi1: {jnp.min(psi1)}, mean psi1: {jnp.mean(psi1)}")
    #R2_2 = jax.vmap(euler_zyz)(phi2, theta2, psi2)
    R2_2 = jax.vmap(lambda phi_, theta_, psi_: euler_zyz(phi_, theta_, psi_, degrees=False))(phi2, theta2, psi2)
    assert jnp.allclose(R1, R1_2, atol=1e-9)
    assert jnp.allclose(R2, R2_2, atol=1e-9)

if __name__ == "__main__":
    pytest.main([__file__])