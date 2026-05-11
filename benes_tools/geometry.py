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