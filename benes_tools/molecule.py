"""
molecule.py — rigid-molecule parser for grid/orientation sampling.

Reads the canonical YAML schema (one entry per interaction site) into
JAX arrays, validates consistency, and exposes boolean masks so the
caller can hand only the relevant sites to the LJ and Ewald kernels.

Runs identically on CPU and GPU: JAX picks the backend automatically.
No GPU is required — the arrays are tiny (3-5 sites per molecule); the
heavy lifting happens later in the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml


# Energy-unit conversions into kelvin (epsilon/k_B), the TraPPE convention.
# Add more as needed; the YAML 'units.energy' field selects the factor.
_BOLTZMANN_K_PER_KJ_MOL = 120.272   # 1 kJ/mol / k_B  -> K
_ENERGY_TO_KELVIN = {
    "K": 1.0,
    "kelvin": 1.0,
    "kJ/mol": _BOLTZMANN_K_PER_KJ_MOL,
    "kcal/mol": _BOLTZMANN_K_PER_KJ_MOL * 4.184,
}


@dataclass
class Molecule:
    """A rigid molecule defined in its body frame.

    All array attributes are length n_segments and index-aligned, so
    mask_lj / mask_coulomb select the same rows across every property.

    Attributes
    ----------
    name, force_field, reference : str
        Metadata carried through from the file.
    combining_rule : str
        LJ cross-term rule ('lorentz-berthelot', 'geometric', ...).
    labels : tuple[str, ...]
        Per-segment site labels (static; not a JAX array).
    positions : (n_segments, 3) jnp.ndarray
        Body-frame coordinates in angstrom.
    sigma, epsilon : (n_segments,) jnp.ndarray
        LJ params; epsilon stored in kelvin. Non-LJ sites hold NaN
        (NaN can't silently leak into a sum the way 0.0 can).
    charge : (n_segments,) jnp.ndarray
        Partial charges in e; non-Coulomb sites are exactly 0.0.
    mask_lj, mask_coulomb : (n_segments,) jnp.ndarray of bool
        True where the site participates in that interaction.
    """

    name: str
    force_field: str
    reference: str
    combining_rule: str
    labels: tuple[str, ...]
    positions: jnp.ndarray
    sigma: jnp.ndarray
    epsilon: jnp.ndarray
    charge: jnp.ndarray
    mask_lj: jnp.ndarray
    mask_coulomb: jnp.ndarray

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Molecule":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Molecule":
        meta = data["metadata"]
        segments = data["segments"]

        # --- unit handling: convert epsilon to kelvin once, up front ---
        units = meta.get("units", {})
        energy_unit = units.get("energy", "K")
        if energy_unit not in _ENERGY_TO_KELVIN:
            raise ValueError(
                f"Unknown energy unit {energy_unit!r}; "
                f"known: {sorted(_ENERGY_TO_KELVIN)}"
            )
        eps_factor = _ENERGY_TO_KELVIN[energy_unit]
        length_unit = units.get("length", "angstrom")
        if length_unit not in ("angstrom", "A", "Angstrom"):
            raise ValueError(
                f"Expected positions in angstrom, got {length_unit!r}. "
                "Convert in the file or extend the loader."
            )

        n_declared = meta.get("n_segments", len(segments))
        if n_declared != len(segments):
            raise ValueError(
                f"metadata.n_segments={n_declared} but found "
                f"{len(segments)} segment entries."
            )

        # --- pull each column, sorting by id for a deterministic order ---
        segments = sorted(segments, key=lambda s: s["id"])
        labels, pos, sig, eps, chg, m_lj, m_cb = [], [], [], [], [], [], []

        for s in segments:
            lj = bool(s["lj"])
            cb = bool(s["coulomb"])
            sigma = s["sigma"]
            epsilon = s["epsilon"]
            charge = float(s["charge"])

            # --- validate the booleans against the data (catch typos) ---
            if not lj and (sigma is not None or epsilon is not None):
                raise ValueError(
                    f"Segment {s['id']} ({s.get('label','?')}): lj=False but "
                    f"sigma/epsilon are not null."
                )
            if lj and (sigma is None or epsilon is None):
                raise ValueError(
                    f"Segment {s['id']} ({s.get('label','?')}): lj=True but "
                    f"sigma/epsilon missing."
                )
            if not cb and charge != 0.0:
                raise ValueError(
                    f"Segment {s['id']} ({s.get('label','?')}): coulomb=False "
                    f"but charge={charge} (expected 0.0)."
                )

            labels.append(s.get("label", str(s["id"])))
            pos.append(s["position"])
            sig.append(np.nan if sigma is None else float(sigma))
            eps.append(np.nan if epsilon is None else float(epsilon) * eps_factor)
            chg.append(charge)
            m_lj.append(lj)
            m_cb.append(cb)

        # --- warn on a non-neutral molecule (usually a mistake) ---
        net_charge = float(np.sum(chg))
        if abs(net_charge) > 1e-6:
            import warnings
            warnings.warn(
                f"Molecule {meta.get('name','?')!r} net charge "
                f"{net_charge:+.4f} e (not neutral).",
                stacklevel=2,
            )

        return cls(
            name=meta.get("name", "unnamed"),
            force_field=meta.get("force_field", "unknown"),
            reference=meta.get("reference", ""),
            combining_rule=meta.get("combining_rule", "lorentz-berthelot"),
            labels=tuple(labels),
            positions=jnp.asarray(pos, dtype=jnp.float32),
            sigma=jnp.asarray(sig, dtype=jnp.float32),
            epsilon=jnp.asarray(eps, dtype=jnp.float32),
            charge=jnp.asarray(chg, dtype=jnp.float32),
            mask_lj=jnp.asarray(m_lj, dtype=bool),
            mask_coulomb=jnp.asarray(m_cb, dtype=bool),
        )

    # ----------------------------------------------------------------- #
    # Convenience accessors for the kernels
    # ----------------------------------------------------------------- #
    @property
    def n_segments(self) -> int:
        return int(self.positions.shape[0])

    def lj_sites(self) -> dict[str, jnp.ndarray]:
        """Return only the LJ sites: positions, sigma, epsilon.

        Uses static boolean indexing (host-side mask -> concrete shape),
        so the result has a fixed size known at trace time and is safe to
        pass into a jitted LJ kernel.
        """
        idx = np.asarray(self.mask_lj)  # host-side, gives a static shape
        return {
            "positions": self.positions[idx],
            "sigma": self.sigma[idx],
            "epsilon": self.epsilon[idx],
        }

    def coulomb_sites(self) -> dict[str, jnp.ndarray]:
        """Return only the Coulomb sites: positions, charge."""
        idx = np.asarray(self.mask_coulomb)
        return {
            "positions": self.positions[idx],
            "charge": self.charge[idx],
        }

    def recenter(self, origin: jnp.ndarray | None = None) -> "Molecule":
        """Return a copy translated so `origin` (default: centroid) is at 0.

        The rotation center your sampler uses should sit at the origin;
        call this if the file's coordinates aren't already centered there.
        """
        if origin is None:
            origin = jnp.mean(self.positions, axis=0)
        return Molecule(
            **{**self.__dict__, "positions": self.positions - origin}
        )

    def __repr__(self) -> str:
        # output from print(Molecule(...)) should be concise but informative, not dump all the arrays
        return (
            f"Molecule(name={self.name!r}, ff={self.force_field!r}, "
            f"n_segments={self.n_segments} "
            f"[n_lj={int(jnp.sum(self.mask_lj))}, "
            f"n_coulomb={int(jnp.sum(self.mask_coulomb))}])"
        )


if __name__ == "__main__":
    # quick smoke test against the inline NH3 schema
    import io

    nh3_yaml = """
metadata:
  name: ammonia
  force_field: TraPPE
  reference: "VERIFY against primary paper"
  units: {length: angstrom, energy: K, charge: e}
  combining_rule: lorentz-berthelot
  n_segments: 5
segments:
  - {id: 0, label: N,  position: [0.0, 0.0, 0.0],     sigma: 3.42, epsilon: 185.0, charge: 0.0,    lj: true,  coulomb: false}
  - {id: 1, label: H1, position: [0.94, 0.0, -0.33],  sigma: null, epsilon: null,  charge: 0.41,   lj: false, coulomb: true}
  - {id: 2, label: H2, position: [-0.47, 0.814, -0.33], sigma: null, epsilon: null, charge: 0.41,  lj: false, coulomb: true}
  - {id: 3, label: H3, position: [-0.47, -0.814, -0.33], sigma: null, epsilon: null, charge: 0.41, lj: false, coulomb: true}
  - {id: 4, label: M,  position: [0.0, 0.0, 0.08],    sigma: null, epsilon: null,  charge: -1.23,  lj: false, coulomb: true}
"""
    mol = Molecule.from_dict(yaml.safe_load(io.StringIO(nh3_yaml)))
    print(mol)
    print("LJ sites:     ", mol.lj_sites()["positions"].shape)
    print("Coulomb sites:", mol.coulomb_sites()["positions"].shape)
    print("epsilon (K):  ", mol.epsilon)