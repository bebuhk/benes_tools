from si_units import DEGREES
from pymatgen.io.cif import CifParser
import numpy as np
import pandas as pd
import warnings

def read_cif(path2cif, path2ff='input_data/Forcefield/UFF.dat', verbose=False):
    """Read a CIF file and return lattice, angles, charges, fractional atom coords,
    UFF sigma/epsilon per atom, and total unit-cell mass."""
    cif_parser = CifParser(path2cif, occupancy_tolerance=100)
    structure   = cif_parser.get_structures(primitive=False)[0]

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

    forcefield = pd.read_csv(path2ff, sep=r'\s+', names=['type', 'sigma', 'epsilon', 'mass'])
    mass_atom  = 1.6605402e-27  # kg / amu
    n_atoms    = structure.num_sites
    sigmas   = np.zeros(n_atoms)
    epsilons = np.zeros(n_atoms)
    mass_kg  = 0.0

    for i, site in enumerate(structure):
        if ':' not in site.species_string:
            idx = forcefield['type'] == site.species_string
            sigmas[i]   = float(forcefield['sigma'][idx].iloc[0])
            epsilons[i] = float(forcefield['epsilon'][idx].iloc[0])
            mass_kg    += float(forcefield['mass'][idx].iloc[0]) * mass_atom
        else:
            species, occ = site.species_string.split(':')
            occ = float(occ)
            idx = forcefield['type'] == species
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

    if not np.isnan(charges).any() and abs(np.sum(charges)) > 1e-6:
        warnings.warn(f'Sum of framework charges is not zero: {np.sum(charges):.4f} e')

    if verbose:
        print(f'lattice (rows = a1,a2,a3):\n{lattice}')
        print(f'n_atoms: {n_atoms}')
        print(f'max/min charge: {np.max(charges):.3f} / {np.min(charges):.3f} e')
        print(f'unit-cell volume: {np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2]))):.3f} Å³')

    return lattice, angles, charges, atoms_abc, sigmas, epsilons, mass_kg


# # ── Run ───────────────────────────────────────────────────────────────────────
# lattice, angles, charges, atoms_abc, sigmas_atoms, epsilons_atoms, mass_kg = read_cif(
#     cif_path, path2ff=framework_ff, verbose=True
# )