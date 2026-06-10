"""
bene, 2026-06-03
orientations.py — uniform SO(3) sampling via Super-Fibonacci spirals.

Reference
---------
Alexa, M. (2022). "Super-Fibonacci Spirals: Fast, Low-Discrepancy
Sampling of SO(3)." CVPR 2022, pp. 8291-8300. (Algorithm 1.)

Pipeline:  super_fibonacci(n) -> (n,4) quaternions
           quats_to_matrices(q) -> (n,3,3) rotation matrices  [JAX, jit/vmap]
Validation against scipy.spatial.transform.Rotation.

Quaternion convention used throughout this module: SCALAR-FIRST, i.e.
q = (w, x, y, z) with w = cos(theta/2). scipy uses scalar-LAST (x,y,z,w),
so we reorder explicitly at the scipy boundary -- never implicitly.
"""

########################################################################
# Fibonacci rotations
########################################################################

import jax
import jax.numpy as jnp
import numpy as np
import itertools

# Production runs use float32 (fast, GPU-friendly). Validation wants real
# float64 to verify the math rather than float32 round-off; callers that
# need x64 must set this BEFORE the first jax computation:
#     jax.config.update("jax_enable_x64", True)


# Paper's recommended constants (Eq. 10-11): phi^2 = 2, psi^4 = psi + 4.
_PHI = np.sqrt(2.0)
_PSI = 1.533751168755204288118041  # positive real root of psi^4 - psi - 4


### bene 2026-06-03: sampling SO(3) efficiently -> perfect of cDFT (and GCMC, not sure if they use the same alg though)
def super_fibonacci(n: int, dtype=jnp.float32) -> jnp.ndarray:
    """Generate n low-discrepancy orientations on SO(3) as unit quaternions.
    implemented after Alexa, 2022, Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO(3).
    see also: https://marcalexa.github.io/superfibonacci/

    Returns
    -------
    (n, 4) array, scalar-FIRST (w, x, y, z), each row a unit quaternion.

    Notes
    -----
    Direct vectorisation of Algorithm 1. The paper writes the quaternion
    as (r sin a, r cos a, R sin b, R cos b); we place that 4-vector into
    scalar-first slots consistently (the labelling is arbitrary as long as
    the matrix conversion below matches it -- the validation test enforces
    this).
    """
    i = jnp.arange(n, dtype=jnp.float64 if dtype == jnp.float64 else jnp.float32)
    s = i + 0.5
    t = s / n
    d = 2.0 * jnp.pi * s
    r = jnp.sqrt(t)
    R = jnp.sqrt(1.0 - t)
    alpha = d / _PHI
    beta = d / _PSI
    q = jnp.stack(
        [r * jnp.sin(alpha),   # w
         r * jnp.cos(alpha),   # x
         R * jnp.sin(beta),    # y
         R * jnp.cos(beta)],   # z
        axis=-1,
    )
    return q.astype(dtype)


### bene 2026-06-03: compute the len(q) rotation matrices (3x3) "instantly". one per quaternion/orientation.
@jax.jit
def quats_to_matrices(q: jnp.ndarray) -> jnp.ndarray:
    """Convert scalar-first unit quaternions (..., 4) to matrices (..., 3, 3).

    Standard quaternion->rotation formula for q = (w, x, y, z). Assumes the
    inputs are already normalised (Algorithm 1 produces unit quaternions),
    but renormalises defensively to absorb float error.
    """
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # rows of the rotation matrix
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    R = jnp.stack(
        [jnp.stack([r00, r01, r02], axis=-1),
         jnp.stack([r10, r11, r12], axis=-1),
         jnp.stack([r20, r21, r22], axis=-1)],
        axis=-2,
    )
    return R


##### for validation: get quaternions from euler angles:
def quat_mul(a, b):
    """Hamilton product, scalar-first (w,x,y,z)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)

def _quat_axis(angle, axis):
    """Unit quaternion for rotation `angle` about a principal axis (0=x,1=y,2=z)."""
    h = angle / 2.0
    w, s = jnp.cos(h), jnp.sin(h)
    vec = [0., 0., 0.]
    q = jnp.array([w, 0., 0., 0.]).at[axis + 1].set(s)
    return q

def euler_zyz_to_quat(phi, theta, psi):
    """(phi, theta, psi) z-y-z  ->  scalar-first unit quaternion."""
    qz1 = _quat_axis(phi,   2)   # Rz(phi)
    qy  = _quat_axis(theta, 1)   # Ry(theta)
    qz2 = _quat_axis(psi,   2)   # Rz(psi)
    return quat_mul(quat_mul(qz1, qy), qz2)
####


########################################################################
# Euler rotations
########################################################################
### bene 2026-06-03: this is mostly to validate. from a rotation matrix, one can go back to Euler angles (phi, theta, psi; zyz convention) via a closed-form formula.

# def matrix_to_euler_zyz(R, degrees=False) -> tuple[float, float, float]:
#     """Recover (phi, theta, psi) from a z-y-z rotation matrix.

#     Inverse of euler_zyz(phi, theta, psi) = Rz(phi) @ Ry(theta) @ Rz(psi).
#     theta in [0, pi]; phi, psi in (-pi, pi].
#     """
#     theta = jnp.arccos(jnp.clip(R[2, 2], -1.0, 1.0))
#     phi   = jnp.arctan2(R[1, 2],  R[0, 2])
#     psi   = jnp.arctan2(R[2, 1], -R[2, 0])

#     # watch out for degeneracy at gimbal lock (theta=0 or 180 degrees), where phi and psi are not uniquely defined (only phi+psi is defined -> infenitely many solutions). We choose to set psi=0 in that case, and let phi absorb the full rotation.
#     if degrees:
#         phi = jnp.degrees(phi)
#         theta = jnp.degrees(theta)
#         psi = jnp.degrees(psi)
#     return phi, theta, psi

# import jax.numpy as jnp

# this can handle a stack of n rotation matrices (R shape (n, 3, 3)) 
def matrix_to_euler_zyz(R, degrees=False):
    """Recover (phi, theta, psi) from z-y-z rotation matrices.

    Inverse of euler_zyz(phi, theta, psi) = Rz(phi) @ Ry(theta) @ Rz(psi).
    Accepts a single (3, 3) matrix or a batch (..., 3, 3).

    Returns
    -------
    If R is (3, 3): a tuple (phi, theta, psi) of scalars.
    If R is (..., 3, 3): an array of shape (..., 3) with columns
        [phi, theta, psi].
    theta in [0, pi]; phi, psi in (-pi, pi].

    At gimbal lock (theta = 0 or pi) only phi + psi is defined; we set
    psi = 0 and let phi absorb the full z-rotation.
    """
    R = jnp.asarray(R)
    single = (R.ndim == 2)

    theta = jnp.arccos(jnp.clip(R[..., 2, 2], -1.0, 1.0))

    # generic (non-degenerate) extraction
    phi = jnp.arctan2(R[..., 1, 2],  R[..., 0, 2])
    psi = jnp.arctan2(R[..., 2, 1], -R[..., 2, 0])

    # --- gimbal-lock handling ---------------------------------------
    # Near theta=0: R = Rz(phi+psi); near theta=pi: R = Rz(phi-psi).
    # The off-axis entries used above vanish, so arctan2(0,0) is junk.
    # Resolve from the top-left 2x2 block and assign everything to phi.
    eps = 1e-7
    near0  = jnp.abs(R[..., 2, 2] - 1.0) < eps     # theta ~ 0
    nearpi = jnp.abs(R[..., 2, 2] + 1.0) < eps     # theta ~ pi
    locked = near0 | nearpi

    # phi + psi (theta~0) or phi - psi (theta~pi) live in the 2x2 block
    phi_lock = jnp.where(
        near0,
        jnp.arctan2(R[..., 1, 0], R[..., 0, 0]),   # phi+psi, sum -> phi
        jnp.arctan2(-R[..., 0, 1], R[..., 0, 0]),  # phi-psi, diff -> phi
    )

    phi = jnp.where(locked, phi_lock, phi)
    psi = jnp.where(locked, 0.0, psi)              # psi absorbed into phi

    if degrees:
        phi, theta, psi = jnp.degrees(phi), jnp.degrees(theta), jnp.degrees(psi)

    if single:
        return phi, theta, psi                     # tuple of scalars, as before
    return jnp.stack([phi, theta, psi], axis=-1)   # (..., 3)


def Rz(a, degrees=False):
    if degrees:
        a = jnp.radians(a)
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([  [ c,    -s,     0.],
                        [ s,    c,      0.],
                        [ 0.,   0.,     1.]])

def Ry(a, degrees=False):
    if degrees:
        a = jnp.radians(a)
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([  [ c,    0.,     s],
                        [ 0.,   1.,     0.],
                        [-s,    0.,     c]])

def euler_zyz(phi, theta, psi, degrees=False):
    """Orientation matrix, z-y-z convention
    R = Rz(phi) @ Ry(theta) @ Rz(psi)

    INPUTS:
    phi : float
        angle about the original z axis (0 to 2*pi)
    theta : float
        angle about the intermediate y axis (0 to pi)
    psi : float
        angle about the final z axis (0 to 2*pi)

    OUTPUT:
    R : (3,3) array
        rotation matrix corresponding to the given Euler angles.
    """
    if degrees:
        phi = jnp.radians(phi)
        theta = jnp.radians(theta)
        psi = jnp.radians(psi)
    return Rz(phi) @ Ry(theta) @ Rz(psi) # first rotate around psi (z axis), then theta (y axis), then phi (z axis). (order matters. gimbal lock can occur (if theta=0 or 180 degrees))



####### bene 9.6.26: vincents angles
def get_naive_angles(angle_step=20):
    """naive equidistant sampling of angles phi and theta. this oversamples poles, not uniform on the sphere. not good for cDFT.
    code is copied from vincents 3d_paper-dft repo.

    INPUT:
    angle_step: float (in degrees) step size for phi and theta angles. smaller step -> more angles -> more accurate but slower cDFT.

    OUTPUT:
    all_angles: list of tuples (phi, theta) in degrees. phi in range 0 to 360, theta in range 0 to 180. (theta is the polar angle, phi is the azimuthal angle)
    """
    theta_angles = np.arange(angle_step, 360, angle_step) # comment bene: would be better to go with standart convention for spherical coordinates: i.e. theta in range 0 to pi, phi in range 0 to 2pi. (otherwise jacobian switches sign..)
    phi_angles = np.arange(0, 180, angle_step)
    #num_config_angles = theta_angles.shape[0]*phi_angles.shape[0] + 1
    #grid_1d = self.framework.grid.reshape((-1, 3)) # grid has shape (n_x, n_y, n_z, 3). grid_1d has shape (n_x*n_y*n_z=#gp, 3) (one row for every grid point. each row: x, y, z coordinate)
    all_angles = list(itertools.product(theta_angles, phi_angles)) 
    all_angles.append((0, 0)) ## bene: one could add (360,0) as well (still missing)

    return all_angles