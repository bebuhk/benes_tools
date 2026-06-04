from si_units import DEGREES
from pymatgen.io.cif import CifParser
import numpy as np
import pandas as pd
import warnings

def read_cif(path2cif, path2ff='input_data/Forcefield/UFF.dat', verbose=False, return_species=False):
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

    forcefield = pd.read_csv(path2ff, sep=r'\s+', names=['type', 'sigma', 'epsilon', 'mass'])
    mass_atom  = 1.6605402e-27  # kg / amu
    n_atoms    = structure.num_sites
    sigmas   = np.zeros(n_atoms)
    epsilons = np.zeros(n_atoms)
    atom_types = []
    mass_kg  = 0.0

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

    if return_species:
        return lattice, angles, charges, atoms_abc, sigmas, epsilons, mass_kg, atom_types
    else:
        return lattice, angles, charges, atoms_abc, sigmas, epsilons, mass_kg


# # ── Run ───────────────────────────────────────────────────────────────────────
# lattice, angles, charges, atoms_abc, sigmas_atoms, epsilons_atoms, mass_kg = read_cif(
#     cif_path, path2ff=framework_ff, verbose=True
# )

def get_grid(lattice, grid_points_per_angstrom=2, ceil2power2_bool=False, all3highest_n_bool=False):
    """Build a 3-D grid of xyz coordinates (cell-centred).
    Returns grid_xyz of shape (Na, Nb, Nc, 3) and n_grid = [Na, Nb, Nc]."""

    # L is length of each lattice vector (rows of lattice).
    n_grid = [int(L * grid_points_per_angstrom) for L in np.linalg.norm(lattice, axis=1)]
    if ceil2power2_bool:
        n_grid = [int(2 ** np.ceil(np.log2(n))) for n in n_grid]
    if all3highest_n_bool:
        n_grid = [max(n_grid)] * 3
    i, j, k = np.indices(n_grid)
    grid_xyz = (
        i[..., np.newaxis] / n_grid[0] * lattice[0] + lattice[0] / (2 * n_grid[0])
      + j[..., np.newaxis] / n_grid[1] * lattice[1] + lattice[1] / (2 * n_grid[1])
      + k[..., np.newaxis] / n_grid[2] * lattice[2] + lattice[2] / (2 * n_grid[2])
    )
    return grid_xyz, n_grid


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
def estimate_parameters(charges, cell, eps_total=1e-8, r_cut=None, alpha=None):
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


def minimize_scalar(fun, bracket=None, bounds=None, args=(),
                    method=None, tol=None, options=None):
    """Local minimization of scalar function of one variable.

    Parameters
    ----------
    fun : callable
        Objective function.
        Scalar function, must return a scalar.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.

    bracket : sequence, optional
        For methods 'brent' and 'golden', `bracket` defines the bracketing
        interval and is required.
        Either a triple ``(xa, xb, xc)`` satisfying ``xa < xb < xc`` and
        ``func(xb) < func(xa) and  func(xb) < func(xc)``, or a pair
        ``(xa, xb)`` to be used as initial points for a downhill bracket search
        (see `scipy.optimize.bracket`).
        The minimizer ``res.x`` will not necessarily satisfy
        ``xa <= res.x <= xb``.
    bounds : sequence, optional
        For method 'bounded', `bounds` is mandatory and must have two finite
        items corresponding to the optimization bounds.
    args : tuple, optional
        Extra arguments passed to the objective function.
    method : str or callable, optional
        Type of solver.  Should be one of:

        - :ref:`Brent <optimize.minimize_scalar-brent>`
        - :ref:`Bounded <optimize.minimize_scalar-bounded>`
        - :ref:`Golden <optimize.minimize_scalar-golden>`
        - custom - a callable object (added in version 0.14.0), see below

        Default is "Bounded" if bounds are provided and "Brent" otherwise.
        See the 'Notes' section for details of each solver.

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options.

        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            Set to True to print convergence messages.

        See :func:`show_options()` for solver-specific options.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    minimize : Interface to minimization algorithms for scalar multivariate
        functions
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is the ``"Bounded"`` Brent method if
    `bounds` are passed and unbounded ``"Brent"`` otherwise.

    Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's
    algorithm [1]_ to find a local minimum.  The algorithm uses inverse
    parabolic interpolation when possible to speed up convergence of
    the golden section method.

    Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the
    golden section search technique [1]_. It uses analog of the bisection
    method to decrease the bracketed interval. It is usually
    preferable to use the *Brent* method.

    Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can
    perform bounded minimization [2]_ [3]_. It uses the Brent method to find a
    local minimum in the interval x1 < xopt < x2.

    Note that the Brent and Golden methods do not guarantee success unless a
    valid ``bracket`` triple is provided. If a three-point bracket cannot be
    found, consider `scipy.optimize.minimize`. Also, all methods are intended
    only for local minimization. When the function of interest has more than
    one local minimum, consider :ref:`global_optimization`.

    **Custom minimizers**

    It may be useful to pass a custom minimization method, for example
    when using some library frontend to minimize_scalar. You can simply
    pass a callable as the ``method`` parameter.

    The callable is called as ``method(fun, args, **kwargs, **options)``
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `bracket`, `tol`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  The method
    shall return an `OptimizeResult` object.

    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method. You can find an example in the scipy.optimize tutorial.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Press, W., S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery.
           Numerical Recipes in C. Cambridge University Press.
    .. [2] Forsythe, G.E., M. A. Malcolm, and C. B. Moler. "Computer Methods
           for Mathematical Computations." Prentice-Hall Series in Automatic
           Computation 259 (1977).
    .. [3] Brent, Richard P. Algorithms for Minimization Without Derivatives.
           Courier Corporation, 2013.

    Examples
    --------
    Consider the problem of minimizing the following function.

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    Using the *Brent* method, we find the local minimum as:

    >>> from scipy.optimize import minimize_scalar
    >>> res = minimize_scalar(f)
    >>> res.fun
    -9.9149495908

    The minimizer is:

    >>> res.x
    1.28077640403

    Using the *Bounded* method, we find a local minimum with specified
    bounds as:

    >>> res = minimize_scalar(f, bounds=(-3, -1), method='bounded')
    >>> res.fun  # minimum
    3.28365179850e-13
    >>> res.x  # minimizer
    -2.0000002026

    """
    if not isinstance(args, tuple):
        args = (args,)

    if callable(method):
        meth = "_custom"
    elif method is None:
        meth = 'brent' if bounds is None else 'bounded'
    else:
        meth = method.lower()
    if options is None:
        options = {}

    if bounds is not None and meth in {'brent', 'golden'}:
        message = f"Use of `bounds` is incompatible with 'method={method}'."
        raise ValueError(message)

    if tol is not None:
        options = dict(options)
        if meth == 'bounded' and 'xatol' not in options:
            warn("Method 'bounded' does not support relative tolerance in x; "
                 "defaulting to absolute tolerance.",
                 RuntimeWarning, stacklevel=2)
            options['xatol'] = tol
        elif meth == '_custom':
            options.setdefault('tol', tol)
        else:
            options.setdefault('xtol', tol)

    # replace boolean "disp" option, if specified, by an integer value.
    disp = options.get('disp')
    if isinstance(disp, bool):
        options['disp'] = 2 * int(disp)

    if meth == '_custom':
        res = method(fun, args=args, bracket=bracket, bounds=bounds, **options)
    elif meth == 'brent':
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          fun, bracket, args, **options)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for '
                             'method `bounded`.')
        res = _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        res = _recover_from_bracket_error(_minimize_scalar_golden,
                                          fun, bracket, args, **options)
    else:
        raise ValueError(f'Unknown solver {method}')

    # gh-16196 reported inconsistencies in the output shape of `res.x`. While
    # fixing this, future-proof it for when the function is vectorized:
    # the shape of `res.x` should match that of `res.fun`.
    res.fun = np.asarray(res.fun)[()]
    res.x = np.reshape(res.x, res.fun.shape)[()]
    return res

