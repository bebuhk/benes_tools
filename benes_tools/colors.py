"""Paul Tol's colorblind-friendly color palettes.

Reference: https://personal.sron.nl/~pault/

Qualitative palettes (discrete categories):
    BRIGHT, HIGH_CONTRAST, VIBRANT, MUTED, MEDIUM_CONTRAST, LIGHT, PALE, DARK

Diverging palettes (ordered, two-sided):
    SUNSET, NIGHTFALL, BURD, PRGN

Sequential palettes (ordered, one-sided):
    YLORBR, IRIDESCENT, INCANDESCENT

Rainbow palettes:
    RAINBOW_DISCRETE (23 distinct colors, indexable subsets recommended by Tol)
    SMOOTH_RAINBOW   (continuous, sample with `smooth_rainbow(n)`)

Each palette also exposes a `BAD` / out-of-range color where Tol specified one.

Quick use:
    from benes_tools.colors import BRIGHT, get_cmap
    plt.plot(x, y, color=BRIGHT[0])
    plt.imshow(z, cmap=get_cmap("sunset"))
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Qualitative palettes
# -----------------------------------------------------------------------------

BRIGHT = (
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
)

HIGH_CONTRAST = (
    "#FFFFFF",  # white
    "#DDAA33",  # yellow
    "#BB5566",  # red
    "#004488",  # blue
    "#000000",  # black
)

VIBRANT = (
    "#EE7733",  # orange
    "#0077BB",  # blue
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#CC3311",  # red
    "#009988",  # teal
    "#BBBBBB",  # grey
)

MUTED = (
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
)
MUTED_BAD = "#DDDDDD"

MEDIUM_CONTRAST = (
    "#FFFFFF",  # white
    "#EECC66",  # light yellow
    "#EE99AA",  # light red
    "#6699CC",  # light blue
    "#997700",  # dark yellow
    "#994455",  # dark red
    "#004488",  # dark blue
    "#000000",  # black
)

LIGHT = (
    "#77AADD",  # light blue
    "#EE8866",  # orange
    "#EEDD88",  # light yellow
    "#FFAABB",  # pink
    "#99DDFF",  # light cyan
    "#44BB99",  # mint
    "#BBCC33",  # pear
    "#AAAA00",  # olive
    "#DDDDDD",  # pale grey
)

PALE = (
    "#BBCCEE",  # pale blue
    "#CCEEFF",  # pale cyan
    "#CCDDAA",  # pale green
    "#EEEEBB",  # pale yellow
    "#FFCCCC",  # pale red
    "#DDDDDD",  # pale grey
)

DARK = (
    "#222255",  # dark blue
    "#225555",  # dark cyan
    "#225522",  # dark green
    "#666633",  # dark yellow
    "#663333",  # dark red
    "#555555",  # dark grey
)

# -----------------------------------------------------------------------------
# Diverging palettes
# -----------------------------------------------------------------------------

SUNSET = (
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF",
    "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026",
)
SUNSET_BAD = "#FFFFFF"

NIGHTFALL = (
    "#125A56", "#00767B", "#238F9D", "#42A7C6", "#60BCE9",
    "#9DCCEF", "#C6DBED", "#DEE6E7", "#ECEADA", "#F0E6B2",
    "#F9D576", "#FFB954", "#FD9A44", "#F57634", "#E94C1F", "#D11807", "#A01813",
)
NIGHTFALL_BAD = "#FFFFFF"

BURD = (
    "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
    "#FDDBC7", "#F4A582", "#D6604D", "#B2182B",
)
BURD_BAD = "#FFEE99"

PRGN = (
    "#762A83", "#9970AB", "#C2A5CF", "#E7D4E8", "#F7F7F7",
    "#D9F0D3", "#ACD39E", "#5AAE61", "#1B7837",
)
PRGN_BAD = "#FFEE99"

# -----------------------------------------------------------------------------
# Sequential palettes
# -----------------------------------------------------------------------------

YLORBR = (
    "#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29",
    "#EC7014", "#CC4C02", "#993404", "#662506",
)
YLORBR_BAD = "#888888"

IRIDESCENT = (
    "#FEFBE9", "#FCF7D5", "#F5F3C1", "#EAF0B5", "#DDECBF",
    "#D0E7CA", "#C2E3D2", "#B5DDD8", "#A8D8DC", "#9BD2E1",
    "#8DCBE4", "#81C4E7", "#7BBCE7", "#7EB2E4", "#88A5DD",
    "#9398D2", "#9B8AC4", "#9D7DB2", "#9A709E", "#906388",
    "#805770", "#684957", "#46353A",
)
IRIDESCENT_BAD = "#999999"

INCANDESCENT = (
    "#CEFFFF", "#C6F7D6", "#A2F49B", "#BBE453", "#D5CE04",
    "#E7B503", "#F19903", "#F6790B", "#F94902", "#E40515", "#A80003",
)
INCANDESCENT_BAD = "#888888"

# -----------------------------------------------------------------------------
# Rainbow palettes
# -----------------------------------------------------------------------------

# 23-color discrete rainbow. Tol recommends choosing an ordered subset for n<23
# (see his Fig. 18). The subsets are stored here so you can pick e.g.
# RAINBOW_DISCRETE_SUBSETS[5] for a 5-color rainbow with good separation.
RAINBOW_DISCRETE = (
    "#E8ECFB", "#D9CCE3", "#D1BBD7", "#CAACCB", "#BA8DB4",
    "#AE76A3", "#AA6F9E", "#994F88", "#882E72", "#1965B0",
    "#437DBF", "#5289C7", "#6195CF", "#7BAFDE", "#4EB265",
    "#90C987", "#CAE0AB", "#F7F056", "#F7CB45", "#F6C141",
    "#F4A736", "#F1932D", "#EE8026", "#E8601C", "#E65518",
    "#DC050C", "#A5170E", "#72190E", "#42150A",
)
RAINBOW_DISCRETE_BAD = "#777777"

# Indices into RAINBOW_DISCRETE for each subset size (1..23), per Tol's tables.
RAINBOW_DISCRETE_SUBSETS = {
    1:  (9,),
    2:  (9, 25),
    3:  (9, 17, 25),
    4:  (9, 14, 17, 25),
    5:  (9, 13, 14, 17, 25),
    6:  (9, 13, 14, 16, 17, 25),
    7:  (8, 9, 13, 14, 16, 17, 25),
    8:  (8, 9, 13, 14, 16, 17, 22, 25),
    9:  (8, 9, 13, 14, 16, 17, 19, 22, 25),
    10: (8, 9, 13, 14, 16, 17, 19, 22, 24, 25),
    11: (3, 6, 8, 9, 13, 14, 16, 17, 19, 22, 25),
    12: (3, 6, 8, 9, 13, 14, 16, 17, 19, 22, 24, 25),
    # Subsets 13-22 follow the same pattern from Tol's technical note; for
    # n>12 it's usually better to use SMOOTH_RAINBOW. The full discrete set
    # remains available via RAINBOW_DISCRETE.
}


def rainbow_discrete(n: int) -> tuple[str, ...]:
    """Return Tol's recommended n-color discrete rainbow subset (1<=n<=12)."""
    if n in RAINBOW_DISCRETE_SUBSETS:
        return tuple(RAINBOW_DISCRETE[i] for i in RAINBOW_DISCRETE_SUBSETS[n])
    if 1 <= n <= len(RAINBOW_DISCRETE):
        # Fallback: evenly spaced indices.
        idx = np.linspace(0, len(RAINBOW_DISCRETE) - 1, n).round().astype(int)
        return tuple(RAINBOW_DISCRETE[i] for i in idx)
    raise ValueError(f"n must be between 1 and {len(RAINBOW_DISCRETE)}")


def smooth_rainbow(n: int) -> tuple[str, ...]:
    """Sample n colors from Tol's smooth (continuous) rainbow.

    The smooth rainbow is defined by analytic R/G/B polynomials in x in [0, 1].
    For diverging-like full-range use, Tol recommends x in [0, 1]; to avoid
    the dark purple end, use x in [0.0, 0.9] etc.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.linspace(0.0, 1.0, n)
    r = 0.472 - 0.567 * x + 4.05 * x**2
    r = r / (1.0 + 8.72 * x - 19.17 * x**2 + 14.1 * x**3)
    g = (
        0.108932
        - 1.22635 * x
        + 27.284 * x**2
        - 98.577 * x**3
        + 163.3 * x**4
        - 131.395 * x**5
        + 40.634 * x**6
    )
    b = 1.0 / (1.97 + 3.54 * x - 68.5 * x**2 + 243.0 * x**3
              - 297.0 * x**4 + 125.0 * x**5)
    rgb = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    return tuple(
        "#{:02X}{:02X}{:02X}".format(int(round(v[0] * 255)),
                                     int(round(v[1] * 255)),
                                     int(round(v[2] * 255)))
        for v in rgb
    )


SMOOTH_RAINBOW_BAD = "#666666"


# -----------------------------------------------------------------------------
# Registry + matplotlib colormap helper
# -----------------------------------------------------------------------------

PALETTES: dict[str, tuple[str, ...]] = {
    "bright": BRIGHT,
    "high_contrast": HIGH_CONTRAST,
    "vibrant": VIBRANT,
    "muted": MUTED,
    "medium_contrast": MEDIUM_CONTRAST,
    "light": LIGHT,
    "pale": PALE,
    "dark": DARK,
    "sunset": SUNSET,
    "nightfall": NIGHTFALL,
    "burd": BURD,
    "prgn": PRGN,
    "ylorbr": YLORBR,
    "iridescent": IRIDESCENT,
    "incandescent": INCANDESCENT,
    "rainbow_discrete": RAINBOW_DISCRETE,
}

BAD_COLORS: dict[str, str] = {
    "muted": MUTED_BAD,
    "sunset": SUNSET_BAD,
    "nightfall": NIGHTFALL_BAD,
    "burd": BURD_BAD,
    "prgn": PRGN_BAD,
    "ylorbr": YLORBR_BAD,
    "iridescent": IRIDESCENT_BAD,
    "incandescent": INCANDESCENT_BAD,
    "rainbow_discrete": RAINBOW_DISCRETE_BAD,
    "smooth_rainbow": SMOOTH_RAINBOW_BAD,
}


def get_cmap(name: str, n: int | None = None):
    """Return a matplotlib colormap from a Tol palette.

    Diverging/sequential palettes become `LinearSegmentedColormap`s;
    qualitative palettes become `ListedColormap`s.

    Args:
        name: palette name (case-insensitive), or "smooth_rainbow".
        n: if given, resample to n colors (ListedColormap).
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    key = name.lower()
    qualitative = {"bright", "high_contrast", "vibrant", "muted",
                   "medium_contrast", "light", "pale", "dark"}

    if key == "smooth_rainbow":
        colors = smooth_rainbow(n if n is not None else 256)
        cmap = LinearSegmentedColormap.from_list("tol_smooth_rainbow", colors)
    elif key in qualitative:
        colors = PALETTES[key]
        cmap = ListedColormap(colors, name=f"tol_{key}")
    elif key in PALETTES:
        colors = PALETTES[key]
        cmap = LinearSegmentedColormap.from_list(f"tol_{key}", colors,
                                                 N=n if n is not None else 256)
    else:
        raise KeyError(f"unknown palette {name!r}. Available: "
                       f"{sorted(PALETTES)} + 'smooth_rainbow'")

    if key in BAD_COLORS:
        cmap.set_bad(BAD_COLORS[key])
    return cmap


__all__ = [
    "BRIGHT", "HIGH_CONTRAST", "VIBRANT", "MUTED", "MEDIUM_CONTRAST",
    "LIGHT", "PALE", "DARK",
    "SUNSET", "NIGHTFALL", "BURD", "PRGN",
    "YLORBR", "IRIDESCENT", "INCANDESCENT",
    "RAINBOW_DISCRETE", "RAINBOW_DISCRETE_SUBSETS",
    "rainbow_discrete", "smooth_rainbow",
    "PALETTES", "BAD_COLORS", "get_cmap",
]
