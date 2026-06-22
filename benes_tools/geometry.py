import numpy as np


# Spherical coords of each atom in the molecule (used for full sampling rotations)
def xyz2spherical(coords, reverse=False, degrees=False):
    """Convert between Cartesian and spherical coordinates. [N x (x, y, z) -> N x (r, theta, phi)].
    input: 
        coords: array of shape (N, 3) representing N points in either Cartesian (x, y, z) [or spherical (r, theta, phi) coordinates if reverse=True]
        reverse: if False, convert from Cartesian to spherical. If True, convert from spherical to Cartesian.
        degrees: if True, angles (theta, phi) are in degrees. If False, angles are in radians.

    output:
        array of shape (N, 3) representing N points (coords) in the other coordinate system. [default: (r, theta, phi)]
    """
    if not reverse:
        coords = np.copy(coords)
        coords = np.float64(coords)  # ensure it's a numpy array (of floats)

        out = np.zeros_like(coords)
        for i, (x, y, z) in enumerate(coords):
            r = np.sqrt(x*x + y*y + z*z)
            if r == 0:
                out[i] = (0, 0, 0)
            else:
                theta = np.arccos(z / r)
                phi   = np.sign(y) * np.arccos(x / np.sqrt(x*x + y*y)) if (x*x + y*y) > 0 else 0.0
                out[i] = (r, theta, phi)
        if degrees:
            out[:,1:] = np.degrees(out[:,1:])
        return out
    if reverse:
        # convert from spherical to Cartesian (x, y, z) = (r sinθ cosφ, r sinθ sinφ, r cosθ)
        if degrees:
            # input in degrees. convert angles (theta, phi) to radians first
            coords[:,1:] = np.radians(coords[:,1:])
        return np.array([[r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)] for r, theta, phi in coords])


### bene 2026-06-03: sampling the surface of a sphere (orientations) efficiently -> perfect of cDFT for molecules with rotational symmetry (i.e. linear molecules like CO2)
def fibonacci_bene(n_full_sphere):
    """fibonacci sampling for uniform points on the full unit sphere.
    implemented after Gonzalez, J. A. (2010). Measurement of areas on a sphere using Fibonacci and latitude-longitude lattices. Mathematical Geosciences, 42(1), 49-64.
    Input: 
    n_full_sphere
        total number of points on the full sphere (including both hemispheres).

    Returns:
    theta_deg, phi_deg
        arrays of shape (n_full_sphere,) with the spherical coordinates in degrees.
    """
    golden_ratio = (1 + np.sqrt(5.0)) / 2.0
    phi_golden = 2*np.pi * (1 - 1/golden_ratio)  # angle in radians between points in the azimuthal direction

    N = int((n_full_sphere- 1) / 2 + 0.5) # the + 0.5 makes sure that we either gen n points (if n_full_sphere is odd) or n+1 points (if n_full_sphere is even)
    i = np.arange(-N, N+1) # 
    phi = (i % golden_ratio) * 360 / golden_ratio # azimuthal angle in radians
    theta = np.arcsin(2 * i/ (2 * N + 1)) * 180/np.pi  # polar angle in degrees, arcsin maps to [-90, 90]
    return theta + 90, phi


def get_angle_between_vectors(v1, v2, degrees=True):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if not degrees:
        return angle_rad
    angle_deg = np.degrees(angle_rad)
    return angle_deg
