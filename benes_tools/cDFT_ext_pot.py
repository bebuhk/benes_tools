#!/usr/bin/env python3

#### bene 2026-06-04: copied to benes_tools from fc_saft_cdft (fc-dft-light)
"""
Author: Benedikt Buhk
Date: 2025-02-01
Description: This script contains functions to calculate the Ewald potential
"""

import numpy as np
import si_units as si
import itertools
from scipy.optimize import minimize_scalar
from scipy.special import erfc, erf


eps0 = 8.8541878188e-12 * si.COULOMB / si.VOLT / si.METER
factor_coulomb = (
    1 / (4 * np.pi * eps0) * si.QE**2 / si.KB / si.KELVIN / si.ANGSTROM
)  # convert from e**2/Å units to K (-> K/molecule)

import numpy as np

from scipy import special

def get_kvec(lattice, lr_wv=2.291):
    # WRONG for some skewed unit cells!!
    """kvec in abc directions (reverse engineered from RASPA for cutoff=12Å)"""
    norm = np.linalg.norm(lattice, axis=1)
    return list([int(x) for x in (norm + lr_wv - 0.58) / lr_wv])

### bene 15.6.26
def get_N_kvecs_LEGACY(lattice, k_cutoff):
    # WRONG for some skewed unit cells!!
    """Get the number of k-vectors for a given lattice and k_cutoff."""
    B, V = reciprocal_lattice(lattice)
    #bmin = min(np.linalg.norm(B, axis=1))  ## this is bad, takes min for all three k-directions. 
    bmin = np.linalg.norm(B, axis=1) ## better: take N individual for each k-direction
    N_kvecs = np.ceil(k_cutoff / bmin).astype(int) + 1
    #nmax = int(np.ceil(k_cutoff / bmin)) + 1
    #rng = [range(-nmax, nmax+1) if p else range(0, 1) for p in periodic]
    return N_kvecs
### bene 16.6.26
def get_N_kvecs(lattice, k_cutoff):
    # corrected
    """Get the number of k-vectors for a given lattice and k_cutoff."""
    #B, V = reciprocal_lattice(lattice)
    a_norms = np.linalg.norm(lattice, axis=1)            # |a_i|
    N_kvecs = np.ceil(k_cutoff * a_norms / (2*np.pi)).astype(int)
    return list([int(N_kvecs[i]) for i in range(len(N_kvecs))])


def reciprocal_lattice(cell, omit_factor_2pi=False):
    """Return reciprocal lattice vectors b_i (rows) and cell volume V. k = l1 b1 + l2 b2 + l3 b3."""
    a1, a2, a3 = cell
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2*np.pi*np.cross(a2, a3)/V
    b2 = 2*np.pi*np.cross(a3, a1)/V
    b3 = 2*np.pi*np.cross(a1, a2)/V
    if omit_factor_2pi:
        return np.array([b1, b2, b3])/(2*np.pi), abs(V)
    else:
        return np.array([b1, b2, b3]), abs(V)

# bene 15.6.26, from claude: /Users/bene/Code/V_ext/_ewald_cdft_derivation.ipynb
def build_kvectors_LEGACY(cell, k_cut, periodic=(True, True, True), omit_factor_2pi=False, min_per_direction=False):
    # this is wrong! for some skewed cells it has blind spots in the cutoffsphere (in kspace). see /Users/bene/Code/V_ext/_check_uc_replication.ipynb
    """Integer-combination k-vectors with |k| <= k_cut, excluding k=0.

    For non-cubic cells the spherical |k|<=k_cut cutoff (not a per-axis integer cap)
    is what keeps the sampling correct under shear. `periodic=False` on an axis
    forbids non-zero shifts along that reciprocal direction (scaffolding only).
    """
    B, V = reciprocal_lattice(cell, omit_factor_2pi=omit_factor_2pi)
    bmin = min(np.linalg.norm(B, axis=1))
    nmax = int(np.ceil(k_cut/bmin)) + 1
    rng = [range(-nmax, nmax+1) if p else range(0, 1) for p in periodic]
    if min_per_direction:
    # # below slightly more efficient. leads to same ks.
        bmin = np.linalg.norm(B, axis=1)
        nmax = np.ceil(k_cut / bmin).astype(int) + 1
        print(f"nmax={nmax}")
        rng = [range(-nmax[i], nmax[i]+1) for i in range(len(nmax))]
    ks = []
    for l1 in rng[0]:
        for l2 in rng[1]:
            for l3 in rng[2]:
                if l1 == l2 == l3 == 0:
                    continue
                k = l1*B[0] + l2*B[1] + l3*B[2]
                if np.linalg.norm(k) <= k_cut:
                    ks.append(k)
    return np.array(ks) if ks else np.zeros((0, 3)), V

def build_kvectors(cell, k_cut, periodic=(True, True, True), verbose=False):
    """Integer-combination k-vectors with |k| <= k_cut, excluding k=0.
 
    Per-axis enumeration bound n_{i,max} = ceil(k_cut * |a_i| / 2pi). This is the
    rigorous tight bound for any cell shape (cubic, orthorhombic, triclinic,
    arbitrarily sheared), because |a_i| equals the perpendicular height of b_i
    to the plane spanned by (b_j, b_k) — exactly the geometric quantity that
    bounds |l_i| when |k| <= k_cut.
 
    NB: this is NOT the same as ceil(k_cut / |b_i|): the latter undercounts on
    sheared lattices because the b_i are not orthogonal.
    """
    cell = np.asarray(cell, float)
    B, V = reciprocal_lattice(cell)
    a_norms = np.linalg.norm(cell, axis=1)            # |a_i|
    nmax = np.ceil(k_cut * a_norms / (2*np.pi)).astype(int)
    if verbose:
        print(f"nmax={nmax}")
    
 
    rng = [range(-nmax[i], nmax[i]+1) if periodic[i] else range(0, 1) for i in range(3)]
    ks = []
    for l1 in rng[0]:
        for l2 in rng[1]:
            for l3 in rng[2]:
                if l1 == l2 == l3 == 0:
                    continue
                k = l1*B[0] + l2*B[1] + l3*B[2]
                if np.linalg.norm(k) <= k_cut:
                    ks.append(k)
    return (np.array(ks) if ks else np.zeros((0, 3))), V


def ewald_lr_grid(lattice, charges, atoms_abc, grid_xyz, N, alpha: float = 0.341429, verbose: bool = True):
    """the long-range part of the Ewald sum
    INPUT:
    - lattice: the lattice vectors (3x3 matrix, abc vectors as rows)
    - charges: vector with charge of each atom in unitcell (1D array, len=#atoms)
    - atoms_abc: the fractional coordinates of the atoms (n_atoms x 3, i.e. each row has abc coordinates of an atom)
    - grid: the grid points (3D arrays) in xyz coordinates

    - N: the number of grid points in each direction (list of 3 integers)
    - a: the Ewald alpha or smearing parameter (float)

    OUTPUT:
    - V_lr: the long-range part of the Ewald sum at each grid point (3D array, same shape as grid_abc[:,:,:,0]) [K/molecule]
    """
    assert lattice.shape == (3, 3), "lattice must be a 3x3 matrix"
    assert len(charges.shape) == 1, "charges must be a 1D array"
    assert len(atoms_abc.shape) == 2, "atoms_abc must be a 2D array: n_atoms x 3"
    assert (
        atoms_abc.shape[0] == charges.shape[0]
    ), "please provide a charge for each atom (i.e. charges and atoms_abc must have the same length)"
    assert (
        atoms_abc.shape[1] == 3
    ), "atoms_abc must have 3 columns (the abc coordinates)"
    assert grid_xyz.shape[3] == 3, "grid must have 3 columns (the xyz coordinates)"
    assert type(N) == list, "N must be a list of 3 integers"
    assert len(N) == 3, "N must be a list of 3 integers"
    assert (
        type(N[0]) == int and type(N[1]) == int and type(N[2]) == int
    ), "all three N values (x,y,z) must be an integer"

    grid_abc = grid_xyz @ np.linalg.inv(lattice)

    if sum(charges) != 0 and verbose:
        print(f"Warning: the total charge of the system is not zero: {sum(charges)}")

    V_lr = np.zeros_like(grid_abc[:, :, :, 0])
    lattice_inv = np.linalg.inv(lattice)
    volume = np.abs(
        np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    )  # a1 dot (a2 x a3)

    # store in array for speed
    b1 = lattice_inv[:, 0]
    b2 = lattice_inv[:, 1]
    b3 = lattice_inv[:, 2]

    for n1, n2, n3 in itertools.product(
        *[
            list(range(N[0] + 1)),
            list(range(-N[1], N[1] + 1)),
            list(range(-N[2], N[2] + 1)),
        ]
    ):  # note: more efficient with np.linalg.norm(k) <= k_cut.... (bene 15.6.26)
        # 0: n1=0, n2=-4, n3=-4 (for N=4)
        # 1: n1=1, n2=-4, n3=-4
        if n1 == 0 and n2 == 0 and n3 == 0:
            continue
        k = n1 * b1 + n2 * b2 + n3 * b3
        # k = n1*lattice_inv[:,0]+n2*lattice_inv[:,1]+n3*lattice_inv[:,2]
        k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2]
        c = 1 if n1 == 0 else 2
        # c = 2
        for i, q_i in enumerate(charges):  # for each atom i
            V_lr += (
                c
                * q_i
                * np.cos(
                    2
                    * np.pi
                    * (
                        n1 * (grid_abc[:, :, :, 0] - atoms_abc[i, 0])  # a difference
                        + n2 * (grid_abc[:, :, :, 1] - atoms_abc[i, 1])  # b difference
                        + n3 * (grid_abc[:, :, :, 2] - atoms_abc[i, 2])  # c difference
                    )
                )
                * np.exp(-k2 * (np.pi / alpha) ** 2)
                / k2
            )
    return V_lr * factor_coulomb / (volume * np.pi)


def calc_C_sr_ExtPot(
    coords_abc, lattice, grid_xyz, charges, rcut=12, alpha=0.341429, inf_mask=None
) -> np.ndarray:
    """_calculates the sr part of the Ewald sum [K/molecule]_

    :param coords_abc: _description_
    :type coords_abc: _type_
    :param lattice: _description_
    :type lattice: _type_
    :param grid_xyz: _description_
    :type grid_xyz: _type_
    :param charges: _description_
    :type charges: _type_
    :param rcut: _description_, defaults to 12
    :type rcut: int, optional
    :param alpha: _description_, defaults to 0.341429
    :type alpha: float, optional
    :param inf_mask: _here, a mask for entries that dont have to be calculated and instead be set to np.inf
    (e.g. because its above max (cutoff) energy anyways as these gridpoints are close to an atom or part of a blocked pore) can
    be handed over. then, only gridpoints where the mask is 0 will be calculated to gain speed._, defaults to None
    :type inf_mask: _type_, optional
    :return: _description_
    :rtype: np.ndarray
    """
    coords_xyz = coords_abc @ lattice  # linear transformation from abc to xyz

    rep, rep_lat = compute_num_images(lattice, rcut)
    num_atoms = len(coords_abc)
    rep_num_atoms = num_atoms * rep[0] * rep[1] * rep[2]  # repeats unitcell in box

    periodic_coordinates = np.zeros([rep_num_atoms, 3])
    periodic_charges = np.zeros([rep_num_atoms])
    # periodic_sigma = np.zeros([rep_num_atoms])
    # periodic_epsilon = np.zeros([rep_num_atoms])
    count = 0
    for i in range(rep[0]):
        for j in range(rep[1]):
            for k in range(rep[2]):
                for a in range(num_atoms):
                    image = i * lattice[0] + j * lattice[1] + k * lattice[2]
                    periodic_coordinates[count] = coords_xyz[a] + image
                    periodic_charges[count] = charges[a]
                    # periodic_sigma[count] = sigmas[a]
                    # periodic_epsilon[count] = epsilons[a]
                    # periodic_type.append(structure_adsorbent.as_dict()["sites"][a]["species"][0]["element"])
                    count += 1

    coulomb_sr = np.zeros(grid_xyz.shape[0:3])
    if inf_mask is not None:
        coulomb_sr[inf_mask] = np.inf

    # coords
    for i in range(grid_xyz.shape[0]):
        for j in range(grid_xyz.shape[1]):
            for k in range(grid_xyz.shape[2]):
                if coulomb_sr[i, j, k] != np.inf:
                    ## DOUBLE CHECK THIS PART (bene 15.12.24) -> verified 14.01.2025
                    distances_coul = (
                        periodic_coordinates - grid_xyz[i, j, k]
                    ) @ np.linalg.inv(
                        rep_lat
                    )  # (3,3) matrix (all < 1)
                    distances_coul = (
                        distances_coul
                        - np.rint(distances_coul)  # round to nearest integer (WHY?)
                    ) @ rep_lat  # (3,3) matrix
                    distances_coul = np.linalg.norm(
                        distances_coul, axis=-1
                    )  # distances to all the (unit-cell-)atoms in a distace of 70 Å or less
                    below_threshold_coul = np.where(distances_coul < rcut)[
                        0
                    ]  # distances to all the (unit-cell-)atoms in a distace of 30 Å or less

                    charge_prod = periodic_charges[
                        below_threshold_coul
                    ]  # charge of segment times vector of charges of all atoms in <= 30 Å distance
                    all_coul = (
                        charge_prod
                        * special.erfc(0.341429 * distances_coul[below_threshold_coul])
                        / distances_coul[below_threshold_coul]
                    )
                    coulomb_sr[i, j, k] += (
                        np.sum(all_coul) * factor_coulomb
                    )  ## Bene 14.01.2025: verified. this is the correct coulomb interaction calculation in [K/molecule]
    return coulomb_sr  # , periodic_coordinates, periodic_charges

## bene 9.6.26: added the center_grid version from /Users/bene/Code/V_ext/__external_potential_vinc_improved_claude_ewald.ipynb
def calc_C_sr_ExtPot_c(coords_abc, lattice, grid_xyz, charges, rcut=12.0, alpha=0.341429, inf_mask=None, damping=True, center_grid=None):
    """Short-range Ewald contribution at every grid point.
    Returns V_sr [K/e] with shape grid_xyz.shape[:3].
    Grid points where inf_mask is True are skipped and set to +inf.
    the point charges around a cutoff sphere with radius rcut around center_grid (if provided) are considered (otherwise around grid_xyz).
    """

    coulomb_sr  = np.zeros(grid_xyz.shape[:3])
    if rcut <= 0.0:
        print(f"Cutoff radius was choosen smaller or equal 0.0 (rcut={rcut:.4f}). -> returning zeros for the short range Coulombic part of the external potential.")
        return coulomb_sr

    coords_xyz   = coords_abc @ lattice
    rep, rep_lat, _ = compute_num_images(lattice, rcut, return_inv=True)
    n_atoms      = len(coords_abc)
    n_rep        = rep[0] * rep[1] * rep[2]

    periodic_coords  = np.zeros((n_atoms * n_rep, 3))
    periodic_charges = np.zeros(n_atoms * n_rep)
    count = 0
    for ii in range(rep[0]):
        for jj in range(rep[1]):
            for kk in range(rep[2]):
                for a in range(n_atoms):
                    periodic_coords[count]  = coords_xyz[a] + ii*lattice[0] + jj*lattice[1] + kk*lattice[2]
                    periodic_charges[count] = charges[a]
                    count += 1

    rep_lat_inv = np.linalg.inv(rep_lat)

    if inf_mask is not None:
        coulomb_sr[inf_mask] = np.inf

    for i in range(grid_xyz.shape[0]):
        for j in range(grid_xyz.shape[1]):
            for k in range(grid_xyz.shape[2]):
                if coulomb_sr[i, j, k] == np.inf:
                    continue
                d = (periodic_coords - grid_xyz[i, j, k]) @ rep_lat_inv
                d = (d - np.rint(d)) @ rep_lat      # minimum-image
                d = np.linalg.norm(d, axis=-1)
                within = d < rcut

                if center_grid is not None:
                    d2 = (periodic_coords - center_grid[i, j, k]) @ rep_lat_inv
                    d2 = (d2 - np.rint(d2)) @ rep_lat      # minimum-image
                    d2 = np.linalg.norm(d2, axis=-1)
                    within = d2 < rcut
                    
                if damping:
                    coulomb_sr[i, j, k] = factor_coulomb * np.sum(
                        periodic_charges[within] * special.erfc(alpha * d[within]) / d[within]
                    )
                else:
                    coulomb_sr[i, j, k] = factor_coulomb * np.sum(
                        periodic_charges[within] / d[within]
                    )

    return coulomb_sr

def calc_LJ(
    coords_abc,
    lattice,
    grid_xyz,
    sigmas,
    epsilons,
    sigma_segment,
    epsilon_segment,
    rcut=12,
    tail_correction_bool=True,
    inf_mask=None,
    center_grid=None,
) -> np.ndarray:
    """_calculates the LJ potential [K/molecule] on a grid_

    :param coords_abc: _description_
    :type coords_abc: _type_
    :param lattice: _description_
    :type lattice: _type_
    :param grid_xyz: _description_
    :type grid_xyz: _type_
    :param sigmas: _description_
    :type sigmas: _type_
    :param epsilons: _description_
    :type epsilons: _type_
    :param sigma_segment: _description_
    :type sigma_segment: _type_
    :param epsilon_segment: _description_
    :type epsilon_segment: _type_
    :param rcut: _description_, defaults to 12
    :type rcut: int, optional
    :param tail_correction_bool: _description_, defaults to True
    :type tail_correction_bool: bool, optional
    :param inf_mask: _here, a mask for entries that dont have to be calculated and instead be set to np.inf
    (e.g. because its above max (cutoff) energy anyways as these gridpoints are close to an atom or part of a blocked pore) can
    be handed over. then, only gridpoints where the mask is 0 will be calculated to gain speed._, defaults to None
    :type inf_mask: _type_, optional
    :return: 
    - LJ: the LJ potential at each grid point (3D array, same shape as grid_xyz[:,:,:,0]) in K
    - tail_correction: the tail correction to be added to the LJ potential (float, in K)
    :rtype: np.ndarray
    """

    coords_xyz = coords_abc @ lattice  # linear transformation from abc to xyz

    rep, rep_lat = compute_num_images(lattice, rcut)
    num_atoms = len(coords_abc)
    rep_num_atoms = num_atoms * rep[0] * rep[1] * rep[2]  # repeats unitcell in box

    periodic_coordinates = np.zeros([rep_num_atoms, 3])
    # periodic_charges = np.zeros([rep_num_atoms])
    periodic_sigma = np.zeros([rep_num_atoms])
    periodic_epsilon = np.zeros([rep_num_atoms])
    count = 0
    for i in range(rep[0]):
        for j in range(rep[1]):
            for k in range(rep[2]):
                for a in range(num_atoms):
                    image = i * lattice[0] + j * lattice[1] + k * lattice[2]
                    periodic_coordinates[count] = coords_xyz[a] + image
                    # periodic_charges[count] = charges[a]
                    periodic_sigma[count] = sigmas[a]
                    periodic_epsilon[count] = epsilons[a]
                    # periodic_type.append(structure_adsorbent.as_dict()["sites"][a]["species"][0]["element"])
                    count += 1

    periodic_epsilon_mix_LJ = np.sqrt(periodic_epsilon * epsilon_segment)
    periodic_sigma_mix_LJ = (periodic_sigma + sigma_segment) / 2

    LJ = np.zeros(grid_xyz.shape[0:3])
    if inf_mask is not None:
        LJ[inf_mask] = np.inf

    for i in range(grid_xyz.shape[0]):
        for j in range(grid_xyz.shape[1]):
            for k in range(grid_xyz.shape[2]):
                if LJ[i, j, k] != np.inf:
                    ## DOUBLE CHECK THIS PART (bene 15.12.24) -> verified 14.01.2025
                    distances_LJ = (
                        periodic_coordinates - grid_xyz[i, j, k]
                    ) @ np.linalg.inv(
                        rep_lat
                    )  # (3,3) matrix (all < 1)
                    distances_LJ = (
                        distances_LJ
                        - np.rint(distances_LJ)  # round to nearest integer (WHY?)
                    ) @ rep_lat  # (3,3) matrix
                    distances_LJ = np.linalg.norm(
                        distances_LJ, axis=-1
                    )  # distances to all the (unit-cell-)atoms in a distace of 70 Å or less

                    if center_grid is not None:
                        d2 = (periodic_coordinates - center_grid[i, j, k]) @ np.linalg.inv(rep_lat)
                        d2 = (d2 - np.rint(d2)) @ rep_lat      # minimum-image
                        d2 = np.linalg.norm(d2, axis=-1)
                        #within = d2 < rcut
                        below_threshold_LJ = np.where(d2 < rcut)[0]  # distances to all the (unit-cell-)atoms in a distace of 30 Å or less
                    else: 
                        below_threshold_LJ = np.where(distances_LJ < rcut)[
                            0
                        ]  # distances to all the (unit-cell-)atoms in a distace of 30 Å or less

                    all_eps = periodic_epsilon_mix_LJ[below_threshold_LJ]
                    all_sig = periodic_sigma_mix_LJ[below_threshold_LJ]
                    all_dis = distances_LJ[below_threshold_LJ]

                    all_LJ = (
                        4
                        * all_eps
                        * ((all_sig / all_dis) ** 12 - (all_sig / all_dis) ** 6)
                    )  # individual LJ interactions [K / molecule] of all atoms in <= 12 Å distance
                    LJ[i, j, k] += np.sum(
                        all_LJ
                    )  ## Bene 14.01.2025: verified. this is the correct LJ interaction calculation in [K/molecule]

    epsilons_mix_LJ = np.sqrt(epsilons * epsilon_segment)
    sigmas_mix_LJ = (sigmas + sigma_segment) / 2

    if tail_correction_bool:
        # this formula is from the RASPA 2 documentation. In there, V is not clearly defined.
        # maybe it should be the volume of the supercell (i.e. the rep_lat matrix) instead of the unit cell?
        # or maybe the volume of the sphere with radius rcut?
        # it would make more sense then the base cell volume, which one could arbitrarily change.
        # this calculation (with the base cell volume and an additional factor of 2 thats not in the RASPA docu)
        # was reverse engineered by vincent from RASPA 2 calculations (up to small deviations) and worked well in the pcfsaft cdft (see paper).
        volume_cell = np.abs(
            np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
        )  # a1 dot (a2 x a3) [Å^3]
        tail_corrections = (
            2
            * np.pi
            / volume_cell
            * 4
            / 3
            * epsilons_mix_LJ
            * sigmas_mix_LJ**3
            * (1 / 3 * (sigmas_mix_LJ / rcut) ** 9 - (sigmas_mix_LJ / rcut) ** 3)
        )  # array with tail correction due to each atom. [K / molecule]

        tail_correction = 2 * sum(tail_corrections)
    else:
        tail_correction = 0.0

    return LJ, tail_correction

def tail_single_interaction(epsilon_ij, sigma_ij, volume_cell, rcut=14.0):
    r"""Compute the tail correction for a single pair interaction in a periodic system.
    
    Parameters
    ----------
    epsilon_ij : float
        Well depth :math:`\varepsilon_{ab}/k_B` of the pair, in K.
    sigma : float
        Size parameter :math:`\sigma_{ab}` of the pair, in angstrom.
    volume_cell : float
        Volume of the unit cell, in angstrom^3.
    r_cut : float
        Van der Waals cutoff radius :math:`r_c`, in angstrom. Must be the
        same cutoff used in the simulation the result is compared against.

    Returns
    -------
    float
        Integral :math:`I(\sigma_{ab}, \varepsilon_{ab})` in K .
        Negative for any physical LJ pair with ``r_cut`` beyond the minimum,
        since the attractive branch dominates past the cutoff.

    Notes
    -----
    .. math::
    computes 

        I(\sigma_{ab}, \varepsilon_{ab})
        = \int_{r_c}^{\infty} r^2\, u_{ab}^{\mathrm{LJ}}(r)\, \mathrm{d}r
        = \frac{8}{3}\pi\varepsilon_{ab}\sigma_{ab}^{3}
          \left[\frac{1}{3}\left(\frac{\sigma_{ab}}{r_c}\right)^{9}
                - \left(\frac{\sigma_{ab}}{r_c}\right)^{3}\right]
        so that 
        U_{tail,RASPA} = \sum_i^N \sum_j^N \cdot I(\sigma_{ij}, \varepsilon_{ij})
        and 
        U_{tail,s-f} = 2 \cdot \sum_i^{N_s} \sum_j^{N_f} \cdot I(\sigma_{ij}, \varepsilon_{ij})
        with N = N_s + N_f.

    """
    return (
        2* np.pi / volume_cell* 4/ 3
        * epsilon_ij* sigma_ij**3
        * (1 / 3 * (sigma_ij / rcut) ** 9 - (sigma_ij / rcut) ** 3)
    )

def compute_U_tail_for_set_s_f(set_epsilons_s, set_sigmas_s, set_epsilons_f, set_sigmas_f, volume_cell, rcut=14.0):
    """computes U_tail_s-f = 2 * sum_i^N_s sum_j^N_f I(sigma_ij, epsilon_ij) with N = N_s + N_f."""
    assert len(set_epsilons_s) == len(set_sigmas_s), "set_epsilons_s and set_sigmas_s must have the same length."
    assert len(set_epsilons_f) == len(set_sigmas_f), "set_epsilons_f and set_sigmas_f must have the same length."
    U_tail = 0
    for i in range(len(set_sigmas_s)):
        for j in range(len(set_sigmas_f)):
            U_tail += tail_single_interaction(epsilon_ij=np.sqrt(set_epsilons_s[i] * set_epsilons_f[j]), sigma_ij=(set_sigmas_s[i] + set_sigmas_f[j]) / 2, volume_cell=volume_cell, rcut=rcut)
    return 2 * U_tail

def get_inf_mask_close2atom(
    coords_abc,
    lattice,
    grid_xyz,
    distance2atom=2.0,
) -> np.ndarray:
    """_extracts gridpoints that are closer than distance2atom [Å] to any atom_

    :param coords_abc: _description_
    :type coords_abc: _type_
    :param lattice: _description_
    :type lattice: _type_
    :param grid_xyz: _description_
    :type grid_xyz: _type_
    :param distance2atom: _description_, defaults to 2.0
    :type distance2atom: float, optional
    # :return: _returns external_potential_mask in same shape as grid_xyz with all entries set to 0.0 except the ones
    #   closer than distance2atom to any atom of the sorbent, these are set to np.inf (this allows other functions to
    #   skip calculating LJ or coulomb potentials at these gridpoints, as the total potential will be above the
    #   max_energy (cutoff) anyways._
    # :rtype: np.ndarray
    :return: _returns inf_mask: a mask for entries that dont have to be calculated and instead be set to np.inf because these gridpoints are close to an atom_
    :rtype: np.ndarray (shape = grid_xyz.shape[0:3])
    """

    if distance2atom == 0.0:
        return np.zeros(grid_xyz.shape[0:3])

    coords_xyz = coords_abc @ lattice  # linear transformation from abc to xyz

    rep, rep_lat = compute_num_images(lattice, distance2atom)
    num_atoms = len(coords_abc)
    rep_num_atoms = num_atoms * rep[0] * rep[1] * rep[2]  # repeats unitcell in box

    # create periodic coordinates
    periodic_coordinates = np.zeros([rep_num_atoms, 3])
    count = 0
    for i in range(rep[0]):
        for j in range(rep[1]):
            for k in range(rep[2]):
                for a in range(num_atoms):
                    image = i * lattice[0] + j * lattice[1] + k * lattice[2]
                    periodic_coordinates[count] = coords_xyz[a] + image
                    count += 1

    external_potential_mask = np.zeros(grid_xyz.shape[0:3])
    for i in range(grid_xyz.shape[0]):
        for j in range(grid_xyz.shape[1]):
            for k in range(grid_xyz.shape[2]):
                distances_gridpoint = (
                    periodic_coordinates - grid_xyz[i, j, k]
                ) @ np.linalg.inv(
                    rep_lat
                )  # (3,3) matrix (all < 1)
                distances_gridpoint = (
                    distances_gridpoint
                    - np.rint(distances_gridpoint)  # round to nearest integer (WHY?)
                ) @ rep_lat  # (3,3) matrix
                distances_gridpoint = np.linalg.norm(distances_gridpoint, axis=-1)
                below_threshold_LJ = np.where(distances_gridpoint < distance2atom)[0]
                if len(below_threshold_LJ) > 0:
                    external_potential_mask[i, j, k] = np.inf
    inf_mask = np.isinf(external_potential_mask)
    return inf_mask


def get_inf_mask_poreblocking(lattice, grid_xyz, path2block, verbose=False):
    """_extracts gridpoints that are inside blocked pores defined by .block file at path2block"""

    # read the block_spheres
    f = open(path2block, "r")
    num_pores = int(f.readline())
    block_spheres = np.zeros([num_pores, 4])  # shape = (num_pores, 4)
    block_spheres_volume_angstrom3_list = []
    for i in range(num_pores):
        line = f.readline()
        words = line.split()
        # print(f"pore {i}: coordinates = {words[0:3]} (type:{type(words[0])}), radius = {words[3]}")
        block_spheres[i, :3] += float(words[0]) * lattice[0]
        block_spheres[i, :3] += float(words[1]) * lattice[1]
        block_spheres[i, :3] += float(words[2]) * lattice[2]
        block_spheres[i, 3] = float(words[3])
        # print(f"blocked_spheres: {block_spheres[i]}")
        block_sphere_volume_angstrom3 = 4 / 3 * np.pi * (float(words[3])) ** 3
        block_spheres_volume_angstrom3_list.append(block_sphere_volume_angstrom3)
    f.close()
    if verbose:
        print(f"block_spheres: {block_spheres}")
        print(
            f"block_spheres_volume_angstrom3_list: {block_spheres_volume_angstrom3_list}"
        )

    external_potential_mask = np.zeros(grid_xyz.shape[0:3])
    for i in range(grid_xyz.shape[0]):
        for j in range(grid_xyz.shape[1]):
            for k in range(grid_xyz.shape[2]):
                for L in range(
                    block_spheres.shape[0]
                ):  # shape = (num_pores, 4) -> for each line/pore L
                    dist = (
                        block_spheres[L, :3]
                        - grid_xyz[i, j, k]  # distance of gridpoints to pore center
                    ) @ np.linalg.inv(lattice)
                    dist = (dist - np.rint(dist)) @ lattice  # mirror distances
                    dist = np.linalg.norm(dist)
                    if dist < block_spheres[L, 3]:  #
                        external_potential_mask[i, j, k] = np.inf
    inf_mask = np.isinf(external_potential_mask)

    if verbose:
        volume_unit_cell_angstrom3 = np.abs(
            np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
        )
        n_gridpoints = np.prod(grid_xyz.shape[0:3])
        n_blocked_gridpoints = np.sum(inf_mask)
        print(
            f"volume_unit_cell_angstrom3: {volume_unit_cell_angstrom3} (Å^3) (n_gridpoints: {n_gridpoints})"
        )
        print(
            f"n_blocked_gridpoints: {n_blocked_gridpoints} ({100*n_blocked_gridpoints/n_gridpoints} %)"
        )
        print(
            f"(MISSLEADING total blocked volume: {np.sum(block_spheres_volume_angstrom3_list)} (Å^3) (sum of block_sphere_volume_angstrom3_list), double counts overlapping gridpoints)"
        )
        print(
            f"total blocked volume: {n_blocked_gridpoints * volume_unit_cell_angstrom3 / n_gridpoints} (Å^3) (n_blocked_gridpoints * volume_unit_cell_angstrom3 / n_gridpoints)"
        )
    return inf_mask, block_spheres_volume_angstrom3_list


def compute_num_images(lattice, cutoff, return_inv=False):
    """_computes the number of images in each direction that is necessary to have the cutoff smaller than the half of distances between faces_"""
    # This function computes the number of images in each direction that is
    # necessary to have the cutoff smaller than the half of distances
    # between faces.
    ws = np.zeros([lattice.shape[0]])
    rep = np.zeros([lattice.shape[0]], dtype=int)
    rep_lat = np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):  # for each direction (a, b, c)
        # Compute the interplanar distance in direction `i`
        cross_prod = np.cross(
            lattice[(i + 1) % 3], lattice[(i + 2) % 3]
        )  # perpendicular onto the other 2 unitvectors
        ws[i] = np.linalg.norm(np.dot(lattice[i], cross_prod)) / np.linalg.norm(
            cross_prod
        )  # the distance between the 2 faces in direction i
        # Determine the number of repetitions needed in direction `i`
        rep[i] = np.ceil(2 * cutoff / ws[i])
        rep_lat[i] = rep[i] * lattice[i]
    # rep_lat_inv = np.linalg.inv(rep_lat)
    if return_inv:
        rep_lat_inv = np.linalg.inv(rep_lat)
        return rep, rep_lat, rep_lat_inv
    return rep, rep_lat

## bene 8.6.26: get the FEA (free-energy-average function) and the canonical average function (to be avoided but used by vincent)
def FEA_Abraham(E_sum_K_na, temperature_K = 298.15):
    """Free-energy average (aka "Boltzmann-averaged" effective interaction) as in Abraham et al. (and Forte2014effective and EllerGross2021FEA)
    INPUT:
        E_sum_K_na: array of shape (Nw), sum of LJ and coulomb contributions for each orientation (Nw is the number of orientations sampled and over which the average should be taken)
        temperature_K: float
    OUTPUT:
        Ew_Abraham: float
    """
    sum_weights = np.sum(np.exp(-E_sum_K_na / temperature_K), axis=0)
    if sum_weights == 0:
        print(f"Warning: sum of weights = 0. Returning Ew_Abraham = np.inf")
        return np.inf
    Ew_Abraham = -temperature_K * np.log(sum_weights/len(E_sum_K_na))
    return Ew_Abraham

def FEA_Abraham_ns(E_sum_K_na, temperature_K=298.15, orientation_sampling_axis=0):
    """numerically stableFree-energy average (aka "Boltzmann-averaged" effective interaction) as in Abraham et al. (and Forte2014effective and EllerGross2021FEA), with numerical stability shift.
        INPUT:
        E_sum_K_na: array of shape (Nw), sum of LJ and coulomb contributions for each orientation (Nw is the number of orientations sampled and over which the average should be taken)
        temperature_K: float
        orientation_sampling_axis: int (axis along which orientations are sampled, typically 0)
    OUTPUT:
        Ew_Abraham: float
    """
    N = len(E_sum_K_na)
    E0 = np.min(E_sum_K_na, axis=orientation_sampling_axis)            # shift for stability
    #print(f"shift by E0 = {E0} K for numerical stability in FEA_Abraham_ns")
    z = np.sum(np.exp(-(E_sum_K_na - E0) / temperature_K), axis=orientation_sampling_axis) / N
    if z == 0:
        print(f"Warning: average of weights Z = 0. Returning Ew_Abraham_ns = np.inf")
        return np.inf
    return E0 - temperature_K * np.log(z)

# Boltzmann-weighted mean over orientations
def canonical_average(energies_K_na, temperature_K = 298.15):
    """canonical average (aka Boltzmann-weighted average) energy over orientations, given energies in K. (as computed in vincents 3d_paper-dft, misleadingly called Boltzmann average in the SI)"""
    w = np.exp(-energies_K_na / temperature_K)
    Z = np.sum(w, axis=0)
    if Z == 0:
        print(f"Warning: sum of weights Z = 0. Returning Ew = np.inf")
        return np.inf, w
    Ew = np.sum(energies_K_na * w, axis=0) / Z
    return Ew, w

def compute_K_H(Vext_K, lattice, T_K, framework_mass_kg):
    """Compute Henry's constant from a grid of external potential energies.

    Estimates :math:`K_H` via Boltzmann-weighted ensemble averaging of the
    external potential over the unit cell grid (Widom insertion method).

    Parameters
    ----------
    Vext_K : ndarray, shape (nx, ny, nz)
        External potential energy at each grid point, in K.
    lattice : ndarray, shape (3, 3), optional
        Lattice vectors of the unit cell, in Å (rows are vectors).
    T_K : float, optional
        Temperature, in K.
    framework_mass_kg : float, optional
        Mass of the framework unit cell, in kg.

    Returns
    -------
    float
        Henry's constant :math:`K_H`, in mol / (Pa kg).

    Notes
    -----
    .. math::

        K_H = \\frac{1}{R T_K} \\cdot \\frac{V_{\\mathrm{cell}}}{m_{\\mathrm{fw}}}
              \\left\\langle e^{-V_{\\mathrm{ext}} / T_K} \\right\\rangle

    where the average is taken over all grid points and :math:`V_{\\mathrm{cell}}`
    is the unit cell volume computed from `lattice`.

    Examples
    --------
    >>> K_H = compute_K_H(Vext_K, lattice, T=300.0, framework_mass_kg=1e-3)

    or 

    >>> K_H = compute_K_H(Vext_K, lattice=lattice, T_K=setup.temperature_K, framework_mass_kg=mass_kg)
    """
    Vext_K_flatten = Vext_K.reshape(-1)
    N = len(Vext_K_flatten)
    R = 8.31446261815324 # J/(mol*K)
    volume_unit_cell_m3 = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2]))) * 10**-30 # convert from Å³ to m³
    sum = np.sum(np.exp(-Vext_K_flatten / T_K))/N # unitless
    K_H_mol_per_Pa_m3 = sum / (R * T_K) # mol / Pa m3
    K_H_mol_per_Pa = K_H_mol_per_Pa_m3 *volume_unit_cell_m3 # convert to mol / Pa
    K_H_mol_per_Pa_kg = K_H_mol_per_Pa / framework_mass_kg # convert to mol / Pa kg
    return K_H_mol_per_Pa_kg

def main():
    pass


if __name__ == "__main__":
    main()
