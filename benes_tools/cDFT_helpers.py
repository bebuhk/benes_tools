from si_units import DEGREES
from pymatgen.io.cif import CifParser
import numpy as np
import pandas as pd
import warnings
from scipy.special import erfc, erf
from scipy.optimize import minimize_scalar

def read_cif(path2cif, path2ff='input_data/Forcefield/UFF.dat', verbose=False, return_species=False, skip_ff=False):
    """Read a CIF file and return lattice, angles, charges, fractional atom coords,
    UFF sigma/epsilon per atom, and total unit-cell mass."""
    cif_parser = CifParser(path2cif, occupancy_tolerance=100)
    #structure   = cif_parser.get_structures(primitive=False)[0]
    structure   = cif_parser.parse_structures(primitive=False)[0] # parse_structures() is equivalent to get_structures(primitive=False)

    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    alpha_rad = structure.lattice.angles[0] * 2 * np.pi / 360
    beta_rad  = structure.lattice.angles[1] * 2 * np.pi / 360
    gamma_rad = structure.lattice.angles[2] * 2 * np.pi / 360
    alpha_deg, beta_deg, gamma_deg = structure.lattice.angles
    angles = [alpha_deg * DEGREES, beta_deg * DEGREES, gamma_deg * DEGREES]

    xi = (np.cos(alpha_rad) - np.cos(gamma_rad) * np.cos(beta_rad)) / np.sin(gamma_rad)
    lat = np.array([
        [a,                 b * np.cos(gamma_rad), c * np.cos(beta_rad)],
        [0,                 b * np.sin(gamma_rad), c * xi],
        [0,                 0,                     c * np.sqrt(1 - np.cos(beta_rad)**2 - xi**2)],
    ])
    lattice = lat.T  # rows = lattice vectors a1, a2, a3

    atoms_abc = structure.frac_coords  # fractional coordinates, shape (n_atoms, 3)

    mass_atom  = 1.6605402e-27  # kg / amu
    n_atoms    = structure.num_sites
    sigmas   = np.zeros(n_atoms)
    epsilons = np.zeros(n_atoms)
    atom_types = []
    mass_kg  = 0.0

    if not skip_ff:
        forcefield = pd.read_csv(path2ff, sep=r'\s+', names=['type', 'sigma', 'epsilon', 'mass'])

        for i, site in enumerate(structure):
            if ':' not in site.species_string:
                idx = forcefield['type'] == site.species_string
                sigmas[i]   = float(forcefield['sigma'][idx].iloc[0])
                epsilons[i] = float(forcefield['epsilon'][idx].iloc[0])
                mass_kg    += float(forcefield['mass'][idx].iloc[0]) * mass_atom
                atom_types.append(site.species_string)
            else:
                species, occ = site.species_string.split(':')
                occ = float(occ)
                idx = forcefield['type'] == species
                atom_types.append(species)
                sigmas[i]   = float(forcefield['sigma'][idx].iloc[0])
                epsilons[i] = float(forcefield['epsilon'][idx].iloc[0]) * occ
                mass_kg    += float(forcefield['mass'][idx].iloc[0]) * mass_atom * occ
                warnings.warn('Partial occupancy detected — handling may be approximate.')

    # Read partial charges from CIF
    if len(cif_parser.as_dict().keys()) != 1:
        warnings.warn('CIF has multiple blocks — charges set to NaN.')
        charges = np.full(len(atoms_abc), np.nan)
    else:
        key = list(cif_parser.as_dict().keys())[0]
        try:
            charges = np.array(cif_parser.as_dict()[key]['_atom_site_charge'], dtype=float)
        except KeyError as e:
            warnings.warn(f'Charge field not found in CIF ({e}). Setting to NaN.')
            charges = np.full(len(atoms_abc), np.nan)

    if not np.isnan(charges).any() and abs(np.sum(charges)) > 1e-8:
        warnings.warn(f'Sum of framework charges is not zero: {np.sum(charges):.8f} e')
    
    volume_unit_cell = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))

    if verbose:
        print(f'lattice (rows = a1,a2,a3):\n{lattice}')
        print(f'n_atoms: {n_atoms} (types: {set(atom_types)})')
        print(f'max and min charge: {np.max(charges):.3f} and {np.min(charges):.3f} e (atoms: {atom_types[np.argmax(charges)]} and {atom_types[np.argmin(charges)]})')
        print(f'unit-cell volume: {volume_unit_cell:.3f} Å³')
        print(f'unit-cell mass: {mass_kg:.3e} kg (-> density: {mass_kg / volume_unit_cell*1e27:.3f} t/m3 = kg/L)')
        #print(f'force field: {path2ff.split("/")[-1]} (sigmas = {set(sigmas)}, epsilons = {set(epsilons)})')

    if skip_ff:
        return lattice, angles, charges, atoms_abc
    elif return_species:
        return lattice, angles, charges, atoms_abc, sigmas, epsilons, mass_kg, atom_types
    else:
        return lattice, angles, charges, atoms_abc, sigmas, epsilons, mass_kg



# # ── Run ───────────────────────────────────────────────────────────────────────
# lattice, angles, charges, atoms_abc, sigmas_atoms, epsilons_atoms, mass_kg = read_cif(
#     cif_path, path2ff=framework_ff, verbose=True
# )

def get_grid(
    lattice,
    grid_points_per_angstrom=2,
    ceil2power2_bool=False,
    all3highest_n_bool=False,
    only_n_grid=False,
):
    """_Creates a grid of xyz coordinates for gridpoints (per unitcell vector direction) evenly spaced in a given lattice._

    :param lattice: _ The lattice vectors in xyz coordinates._
    :type lattice: _np.ndarray_
    :param grid_points_per_angstrom: _The number of grid points per angstrom._, defaults to 2
    :type grid_points_per_angstrom: int, optional
    :param ceil_to_power_of_two: _If True, the grid size is rounded up to the next higher exponent of 2. Defaults to False. This
            can be useful as the FFT algorithm is most efficient for grid sizes that are a power of 2, so the additional grid points are calculated
            without (much) additional computational cost._, defaults to False
    :type ceil_to_power_of_two: bool, optional
    :param all3highest_n_bool: _take the highest of the three n_grids based on the per angstrom (and ceiling) for all three directions_, defaults to False
    :type all3highest_n_bool: bool, optional
    :return: _tuple[grid_xyz: np.ndarray: The grid of xyz coordinates.
        n_grid: list[int]]: The grid size (i.e. the # of grid points) in abc directions. its the same as "list(grid_xyz.shape[0:3])" and outputed here only for conveience._
    :rtype: _type_
    """
    # non orthogonal case ## chat Copilot code

    # get grid size (in abc directions) (2 per angstrom)
    n_grid = [
        int(L * grid_points_per_angstrom) for L in np.linalg.norm(lattice, axis=1)
    ]  # i.e. L in [a,b,c], int always rounds down
    if only_n_grid:
        return n_grid

    # round up to the next higher exponent of 2
    if ceil2power2_bool:
        n_grid = [int(2 ** np.ceil(np.log2(n_grid_i))) for n_grid_i in n_grid]

    if all3highest_n_bool:
        n_grid = [max(n_grid)] * 3

    # n_grid = [33, 33, 33]  # for testing

    # Create a grid of indices
    i, j, k = np.indices(
        n_grid
    )  # blocks, rows, columns [r,c,b] #i,j,k have shape (n_grid[0], n_grid[1], n_grid[2])

    # Calculate the grid points using broadcasting
    grid_xyz = (  # in xyz coordinates
        # np.newaxis creates a new dimension of size 1 -> shape = (n_grid[0], n_grid[1], n_grid[2], 1)
        # which is multiplied with some value in the shape of lattice[x] (shape = (3,))
        # -> the result is a shape of (n_grid[0], n_grid[1], n_grid[2], 3)
        i[..., np.newaxis] / n_grid[0] * lattice[0]
        + lattice[0] / (2 * n_grid[0])  # i is a part of the gridpoint
        + j[..., np.newaxis] / n_grid[1] * lattice[1]
        + lattice[1] / (2 * n_grid[1])  # j is b part of the gridpoint
        + k[..., np.newaxis] / n_grid[2] * lattice[2]
        + lattice[2] / (2 * n_grid[2])  # k is c part of the gridpoint
        # k * step + step/2 (where step = lattice[2]/n_grid[2] = c/#gridpoints_c_direction)
    )
    return grid_xyz, n_grid

def get_gpa_for_n_gridpoints(n_gridpoints, lattice):
    """this function returns the grid points per angstrom (gpa) so that - for the given lattice - the number of grid points in the largest dimension (longest lattice vector) is equal to n_gridpoints. 
    
    This is useful for creating a grid with n_gridpoints in each dimension via

    grid_xyz, n_grid = get_grid(
        lattice,
        grid_points_per_angstrom=get_gpa_for_n_gridpoints(n_gridpoints, lattice),
        ceil2power2_bool=False,
        all3highest_n_bool=True,
    )

    (e.g. getting a grid with 64 grid points in each dimension for a given lattice)
    """

    # get_grid takes int(L * gpa) for L in np.linalg.norm(lattice, axis=1) as n_grid.
    # we only look at the largest L, i.e.
    L_max = np.max(np.linalg.norm(lattice, axis=1))

    # now, to have int(L_max * gpa) = n_gridpoints, we need to solve for gpa:
    gpa = (n_gridpoints+0.5) / L_max # the +0.5 is for numerical stability. (int(64.5) = 64
    return gpa

def get_grid_lattice_atoms(path2cif, grid_points_per_angstrom=2, ceil2power2_bool=False, all3highest_n_bool=False, path2ff=None):
    # bene 29.6.26
    if path2ff is None:
        lattice, angles, charges, atoms_abc = read_cif(path2cif, skip_ff=True)
    else:
        lattice, angles, charges, atoms_abc, sigmas, epsilons, mass_kg = read_cif(path2cif, skip_ff=False, path2ff=path2ff)

    grid_xyz, n_grid = get_grid(lattice, grid_points_per_angstrom=grid_points_per_angstrom, ceil2power2_bool=ceil2power2_bool, all3highest_n_bool=all3highest_n_bool)
    
    if path2ff is None:
        return grid_xyz, lattice, atoms_abc
    else:
        return grid_xyz, lattice, atoms_abc, mass_kg
    
def get_lattice_angles(L: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Return (alpha, beta, gamma) for a lattice with row vectors a, b, c."""
    a, b, c = L
    lengths = np.linalg.norm(L, axis=1)            # |a|, |b|, |c|
    cos_alpha = np.dot(b, c) / (lengths[1] * lengths[2])
    cos_beta  = np.dot(a, c) / (lengths[0] * lengths[2])
    cos_gamma = np.dot(a, b) / (lengths[0] * lengths[1])
    cosines = np.clip([cos_alpha, cos_beta, cos_gamma], -1.0, 1.0)
    angles = np.arccos(cosines)
    return np.degrees(angles) if degrees else angles

def reciprocal_lattice(cell):
    """Return reciprocal lattice vectors b_i (rows) and cell volume V. k = l1 b1 + l2 b2 + l3 b3.
    INPUT: 
    - cell = [a1, a2, a3] with a_i = (x, y, z) rows. 
    
    OUTPUT: 
    - b_i = (x, y, z) rows.
    - V = volume of the unit cell (in Å³ if a_i are in Å).
    
    """
    a1, a2, a3 = cell
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2*np.pi*np.cross(a2, a3)/V
    b2 = 2*np.pi*np.cross(a3, a1)/V
    b3 = 2*np.pi*np.cross(a1, a2)/V
    return np.array([b1, b2, b3]), abs(V)



###########################################################################################################
# Ewald parameter estimation (from Claude, aligned with Frenkel&Smit i think)
def estimate_ewald_parameters(charges, cell, eps_total=1e-8, r_cut=None, alpha=None):
    """Estimate (alpha, r_cut, k_cut) for a target total accuracy.

    Mirrors the reference implementation's logic:
      - split error budget equally between real and reciprocal space
      - if r_cut is fixed, solve for alpha from the real-space estimate
      - then get k_cut from the reciprocal-space estimate
    Returns a dict.
    """
    charges = np.asarray(charges, float)
    Q2 = float(charges @ charges)
    N  = len(charges)
    _, V = reciprocal_lattice(cell)
    eps = eps_total/2

    def real_err(rc, a):  return (Q2/np.sqrt(N))*erfc(a*rc)/rc
    def recip_err(kc, a): return (Q2*a/(np.sqrt(N)*np.pi))*np.exp(-kc**2/(4*a**2))

    if r_cut is not None and alpha is None:
        z = minimize_scalar(lambda z: abs(erfc(z) - r_cut*eps)).x
        alpha = z/r_cut
    elif alpha is None:
        # cost-balanced alpha: minimize rc^3 (real) + nkvec (recip) at fixed accuracy
        Lc = V**(1/3)
        def total_cost(a):
            if a <= 0: return np.inf
            z = minimize_scalar(lambda z: abs(erfc(z) - eps)).x
            rc = z/a
            kc = 2*a*np.sqrt(-np.log(eps))
            n_kvecs = (2*kc*Lc/(2*np.pi) + 1)**3
            rc_cost = rc**3 * N/V * 4/3*np.pi
            return rc_cost + n_kvecs
        grid = np.linspace(0.05, 1.5, 300)
        alpha = grid[np.argmin([total_cost(a) for a in grid])]

    if r_cut is None:
        z = minimize_scalar(lambda z: abs(erfc(z) - eps)).x
        r_cut = z/alpha
    k_cut = 2*alpha*np.sqrt(-np.log(eps))
    return dict(alpha=alpha, r_cut=r_cut, k_cut=k_cut,
                error_real=real_err(r_cut, alpha), error_recip=recip_err(k_cut, alpha))

