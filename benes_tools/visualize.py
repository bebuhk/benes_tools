"""
visualize.py — interactive 3D rendering of a rigid Molecule.

Encoding
--------
- charge  -> diverging color (dark blue = negative, neutral grey at 0,
             dark red = positive), clamped at +/- charge_max (default 2 e).
             Applies to every site; non-Coulomb sites are charge 0 -> neutral.
- sigma   -> sphere radius (= sigma / 2, the LJ contact radius) for LJ sites.
             Non-LJ sites are drawn as a light wireframe sphere at r = 1 A.
- epsilon -> surface opacity of LJ spheres (deeper well = more opaque),
             optional via epsilon_opacity; floored so weak sites stay visible.

Output: an interactive, rotatable Plotly figure (save to standalone HTML).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .molecule import Molecule


# ----------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------- #
# Paul Tol's "sunset" diverging colour scheme (colour-blind safe),
# verified from SRON technical note (personal.sron.nl/~pault).
# Ordered negative -> neutral -> positive; the pale midpoint marks zero.
_TOL_SUNSET = [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF",
    "#EAECCC",  # midpoint (neutral / zero charge)
    "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026",
]


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


_TOL_SUNSET_RGB = np.array([_hex_to_rgb(c) for c in _TOL_SUNSET])

# Same scheme in Plotly colorscale format (position 0..1 -> colour),
# so a colorbar can be drawn that matches _diverging_color exactly.
_TOL_SUNSET_SCALE = [
    [i / (len(_TOL_SUNSET) - 1), c] for i, c in enumerate(_TOL_SUNSET)
]


def _diverging_color(charge: float, charge_max: float) -> str:
    """Map a charge to an 'rgb(...)' string via Tol's sunset scheme.

    charge in [-charge_max, +charge_max] maps onto the full scheme,
    with 0 e at the pale neutral midpoint. Colours are linearly
    interpolated between the 11 anchor colours, as Tol specifies for
    a continuous version of a diverging scheme.
    """
    t = np.clip(charge / charge_max, -1.0, 1.0)          # -1 .. +1
    pos = (t + 1.0) / 2.0 * (len(_TOL_SUNSET_RGB) - 1)    # 0 .. 10
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(_TOL_SUNSET_RGB) - 1)
    frac = pos - lo
    rgb = _TOL_SUNSET_RGB[lo] * (1 - frac) + _TOL_SUNSET_RGB[hi] * frac
    r, g, b = (int(round(v)) for v in rgb)
    return f"rgb({r},{g},{b})"


def _sphere_mesh(center, radius, n=24):
    """Vertices for a UV-sphere; returns x, y, z flat arrays for Mesh3d."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(uu) * np.sin(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(vv)
    return x.flatten(), y.flatten(), z.flatten()


def _wireframe_sphere(center, radius, n=14):
    """Line segments tracing latitude/longitude circles of a sphere."""
    xs, ys, zs = [], [], []
    # latitudes
    for vphi in np.linspace(0, np.pi, n // 2 + 1)[1:-1]:
        t = np.linspace(0, 2 * np.pi, n)
        xs += list(center[0] + radius * np.cos(t) * np.sin(vphi)) + [None]
        ys += list(center[1] + radius * np.sin(t) * np.sin(vphi)) + [None]
        zs += list(center[2] + radius * np.cos(vphi) * np.ones_like(t)) + [None]
    # longitudes
    for vth in np.linspace(0, 2 * np.pi, n, endpoint=False):
        t = np.linspace(0, np.pi, n)
        xs += list(center[0] + radius * np.cos(vth) * np.sin(t)) + [None]
        ys += list(center[1] + radius * np.sin(vth) * np.sin(t)) + [None]
        zs += list(center[2] + radius * np.cos(t)) + [None]
    return xs, ys, zs


def _fmt(value, nan_as="null", prec=3):
    """Format a float, printing `nan_as` for NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return nan_as
    return f"{value:.{prec}g}"


# ----------------------------------------------------------------------- #
# main entry point
# ----------------------------------------------------------------------- #
def visualize_molecule(
    mol: Molecule,
    charge_max: float = 1.0,
    sigma_scale: float = 0.6,
    epsilon_opacity: bool = True,
    opacity_floor: float = 0.25,
    epsilon_min: float = 0.0,
    epsilon_max: float = 150.0,
    coulomb_radius: float = 1.0,
    show_axes: bool = True,
    show_colorbar: bool = True,
    show_opacity_scale: bool = True,
    show_caption: bool = True,
    save_path: str | None = None,  
    cubic_frame: float | None = 3.0,
    fixed_camera: bool = True,
    positions = None,
    title = "",
) -> go.Figure:
    """Build an interactive 3D Plotly figure of `mol`.

    Parameters
    ----------
    mol : Molecule
    charge_max : float
        Charge (e) mapped to the colormap endpoints; clamped beyond +/-.
    sigma_scale : float
        Multiplier on the LJ sphere radius (radius = sigma_scale * sigma / 2).
        Purely visual: shrink (<1) to declutter overlapping sites, or grow
        (>1) to emphasise size differences. Does not affect non-LJ spheres.
    epsilon_opacity : bool
        If True, LJ sphere opacity scales with epsilon. If False, all LJ
        spheres are opaque and the caption omits the opacity explanation.
    opacity_floor : float
        Opacity at epsilon_min (so a shallow-well site stays visible).
    epsilon_min, epsilon_max : float
        Absolute epsilon (K) anchors for the opacity scale: epsilon_min maps
        to opacity_floor, epsilon_max (and above) maps to fully opaque. Fixed
        anchors (default 0 -> 300 K) make opacity comparable across figures
        rather than rescaling per molecule.
    coulomb_radius : float
        Radius (A) of the dashed wireframe spheres marking non-LJ sites.
    show_axes : bool
        If True, draw x/y/z unit vectors (length 1 A) from the origin as
        coloured arrows, labelled at their tips.
    show_colorbar : bool
        If True, draw a colorbar for the charge scale, from -charge_max to
        +charge_max (e), matching the sunset site colours.
    show_opacity_scale : bool
        If True (and epsilon_opacity is on), draw a small legend showing the
        epsilon -> opacity mapping at a few sample values.
    show_caption : bool
        If True, add a journal-style caption below the plot.
    save_path : str or None
        If not None, save a static image (svg, pdf, png) or interactive HTML
        to this path. Static export requires the kaleido package; if not
        available, an HTML file is written instead with a warning.
    cubic_frame : float or None
        If not None, set the 3D axes limits to +/- this value and use a cube aspect ratio,
        so the relative sizing of spheres is accurate and the molecule is centered in a fixed
    fixed_camera : bool
        If True, set a fixed camera position for a more informative default view (side-on,
        with axes labels visible). If False, use Plotly's default auto-rotating camera.)
    positions : array-like or None
        Optional override for the site positions; if None, uses mol.positions.
    title : str
        Optional title prefix to appear before the molecule name in the figure title.
    """
    if positions is not None:
        pos = np.asarray(positions)
    else:
        pos = np.asarray(mol.positions)
    sigma = np.asarray(mol.sigma)
    epsilon = np.asarray(mol.epsilon)
    charge = np.asarray(mol.charge)
    m_lj = np.asarray(mol.mask_lj)

    # epsilon -> opacity, mapped on FIXED absolute anchors so the same
    # epsilon looks identical across molecules/figures.
    eps_span = epsilon_max - epsilon_min
    if epsilon_opacity and eps_span > 0:
        def eps_to_opacity(e):
            frac = np.clip((e - epsilon_min) / eps_span, 0.0, 1.0)
            return opacity_floor + (1.0 - opacity_floor) * frac
    else:
        def eps_to_opacity(e):
            return 1.0

    traces = []
    for i in range(mol.n_segments):
        c = float(charge[i])
        color = _diverging_color(c, charge_max)
        center = pos[i]
        if m_lj[i]:
            radius = sigma_scale * float(sigma[i]) / 2.0
            opac = float(eps_to_opacity(epsilon[i])) if epsilon_opacity else 1.0
            x, y, z = _sphere_mesh(center, radius)
            traces.append(
                go.Mesh3d(
                    x=x, y=y, z=z, alphahull=0,
                    color=color, opacity=opac,
                    name=mol.labels[i], showlegend=False,
                    hovertext=(f"{mol.labels[i]}<br>sigma={_fmt(sigma[i])} A"
                               f"<br>eps={_fmt(epsilon[i])} K"
                               f"<br>q={_fmt(c)} e"),
                    hoverinfo="text",
                )
            )
        else:
            x, y, z = _wireframe_sphere(center, radius=coulomb_radius)
            traces.append(
                go.Scatter3d(
                    x=x, y=y, z=z, mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=mol.labels[i], showlegend=False,
                    hovertext=(f"{mol.labels[i]} (non-LJ, r={coulomb_radius:g} A)"
                               f"<br>q={_fmt(c)} e"),
                    hoverinfo="text",
                )
            )

    # ---- optional xyz unit vectors from the origin ----
    if show_axes:
        axis_specs = [
            ((1, 0, 0), "x", "rgb(200,40,40)"),
            ((0, 1, 0), "y", "rgb(40,160,40)"),
            ((0, 0, 1), "z", "rgb(40,40,200)"),
        ]
        for vec, lbl, col in axis_specs:
            vec = np.asarray(vec, dtype=float)
            shaft_end = vec * 0.80   # shaft stops short; cone forms the hat
            # shaft as a thin line from origin toward the tip
            traces.append(
                go.Scatter3d(
                    x=[0, shaft_end[0]], y=[0, shaft_end[1]], z=[0, shaft_end[2]],
                    mode="lines",
                    line=dict(color=col, width=3),
                    showlegend=False, hoverinfo="skip",
                )
            )
            # arrowhead: a cone spanning the last 20% up to the unit tip
            traces.append(
                go.Cone(
                    x=[vec[0]], y=[vec[1]], z=[vec[2]],
                    u=[vec[0] * 0.22], v=[vec[1] * 0.22], w=[vec[2] * 0.22],
                    sizemode="absolute", sizeref=0.22,
                    anchor="tip", showscale=False,
                    colorscale=[[0, col], [1, col]],
                    hoverinfo="skip",
                )
            )
            # label just beyond the tip
            traces.append(
                go.Scatter3d(
                    x=[vec[0] * 1.25], y=[vec[1] * 1.25], z=[vec[2] * 1.25],
                    mode="text", text=[lbl],
                    textfont=dict(color=col, size=14),
                    showlegend=False, hoverinfo="skip",
                )
            )

    # ---- charge colorbar (a hidden marker trace carrying the scale) ----
    if show_colorbar:
        traces.append(
            go.Scatter3d(
                x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
                mode="markers",
                marker=dict(
                    size=0.1, opacity=0.0,          # invisible point
                    color=[0.0], cmin=-charge_max, cmax=charge_max,
                    colorscale=_TOL_SUNSET_SCALE,
                    colorbar=dict(
                        title=dict(text="charge (e)", side="right"),
                        tickvals=[-charge_max, -charge_max / 2, 0,
                                  charge_max / 2, charge_max],
                        len=0.6, thickness=15, x=0.92,
                    ),
                ),
                showlegend=False, hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    lines = [title + f"<b>{mol.name}</b> ({mol.force_field})"]
    for i in range(mol.n_segments):
        s = _fmt(sigma[i]) if m_lj[i] else "null"
        e = _fmt(epsilon[i]) if m_lj[i] else "null"
        q = _fmt(charge[i]) if mol.mask_coulomb[i] else "null"
        lines.append(
            f"{mol.labels[i]}: \u03c3={s} \u00c5, \u03b5={e} K, q={q} e"
        )
    title = "<br>".join(lines)

    #margins = dict(l=0, r=0, t=20 + 16 * mol.n_segments, b=140 if show_caption else 20)
    margins = dict(l=0, r=0, t=20 + 16 * 5, b=240 if show_caption else 20)
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=12)),
        scene=dict(
            xaxis_title="x (\u00c5)", yaxis_title="y (\u00c5)", zaxis_title="z (\u00c5)",
            aspectmode="data",
        ),
        margin=margins,
        showlegend=False,
    )

    if fixed_camera == True:
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.07),   # low z -> more side-on, axis labels visible
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            )
        )

    # ---- opacity scale legend (epsilon -> opacity), drawn manually ----
    if show_opacity_scale and epsilon_opacity and eps_span > 0:
        samples = [epsilon_max, epsilon_max * 0.5, epsilon_min]
        y0 = 0.34  # top of the legend, paper coords
        dy = 0.10
        fig.add_annotation(
            text="\u03b5 (K) \u2192 [opacity]", xref="paper", yref="paper",
            x=0.865, y=y0 + dy - 0.04, showarrow=False,
            font=dict(size=10), xanchor="center", yanchor="bottom",
        )
        for k, e_val in enumerate(samples):
            op = float(eps_to_opacity(e_val))
            yk = y0 - k * dy
            # grey swatch at the mapped opacity
            fig.add_shape(
                type="rect", xref="paper", yref="paper",
                x0=0.885, x1=0.915, y0=yk - 0.035, y1=yk + 0.035,
                fillcolor="rgb(90,90,90)", opacity=op,
                line=dict(color="rgb(90,90,90)", width=0.5),
            )
            fig.add_annotation(
                text=f"{e_val:g} [{op:.2f}]", xref="paper", yref="paper",
                x=0.875, y=yk, showarrow=False,
                font=dict(size=9), xanchor="right", yanchor="middle",
            )

    if show_caption:
        radius_desc = (
            f"radius = {sigma_scale:g}\u00d7\u03c3/2"
            if sigma_scale != 1.0
            else "radius = \u03c3/2"
        )
        # build as discrete lines joined by <br> so it wraps reliably
        cap_lines = [
            "Site color encodes partial charge on Paul Tol's <i>sunset</i> "
            "diverging scheme (colour-blind safe):",
            f"blue = negative, pale midpoint = 0, red = positive; "
            f"clamped at \u00b1{charge_max:g} e.",
            f"Solid spheres are Lennard-Jones sites with {radius_desc}; "
            f"dashed wireframe spheres (r = {coulomb_radius:g} \u00c5) mark non-LJ sites.",
        ]
        if epsilon_opacity:
            cap_lines.append(
                "Sphere opacity scales with the LJ well depth \u03b5 "
                f"(opacity {opacity_floor:g} at \u03b5 = {epsilon_min:g} K, "
                f"fully opaque at \u03b5 \u2265 {epsilon_max:g} K)."
            )
        else:
            cap_lines.append("All LJ spheres are drawn opaque (\u03b5 not encoded).")
        cap = "<br>".join(cap_lines)

        # bottom margin must grow with the number of caption lines
        margins["b"] = 40 + 18 * len(cap_lines)
        fig.update_layout(margin=margins, width=800, height=600)  # fixed size to keep caption readable

        fig.add_annotation(
            text=cap, xref="paper", yref="paper",
            x=0.5, y=-0.04, showarrow=False,
            font=dict(size=10), align="center",
            xanchor="center", yanchor="top",
        )

    if cubic_frame is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-cubic_frame, cubic_frame]),
                yaxis=dict(range=[-cubic_frame, cubic_frame]),
                zaxis=dict(range=[-cubic_frame, cubic_frame]),
                aspectmode='cube',  # equal-length axes -> correct relative sizing
            )
        )

    if save_path is not None:
        ext = save_path.lower().rsplit(".", 1)[-1]
        if ext in {"svg", "pdf", "png"}:
            try:
                fig.write_image(save_path, width=800, height=600)
            except Exception as e:
                html_path = save_path.rsplit(".", 1)[0] + ".html"
                fig.write_html(html_path)
                print(f"Static export failed ({e}); wrote {html_path} instead. "
                    f"For SVG/PDF: pip install 'plotly>=6.1.1' and run "
                    f"kaleido_get_chrome.")
        elif ext in {"html", "htm"}:
            fig.write_html(save_path)
        else:
            raise ValueError(f"unsupported save_path extension: {ext}")
        print(f"Wrote {save_path}")
            
    # if save_path is not None:
    #     # save as vector image (svg or pdf) for publication
    #     ext = save_path.lower().rsplit(".", 1)[-1]
    #     if ext in {"svg", "pdf", "png"}:
    #         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    #         try:
    #             fig.write_image(save_path, width=600, height=600)
    #         except ValueError as exc:
    #             if "kaleido" in str(exc).lower():
    #                 save_path = str(Path(save_path).with_suffix(".html"))
    #                 fig.write_html(save_path)
    #                 print("kaleido not available; wrote interactive HTML instead.")
    #             else:
    #                 raise
    #     elif ext in {"html", "htm"}:
    #         fig.write_html(save_path)
    #     else: 
    #         raise ValueError(f"unsupported save_path extension: {ext}")
    #     print(f"wrote {save_path}")


    return fig




## bene 12.6.2026: coded this with claude to visualize all SO(3) orientation samples (e.g. from Super-Fibonacci)
"""
plot_so3.py — visualize a set of SO(3) orientations on the unit sphere.

Each orientation R places a base point R@(0,0,1) on the sphere and a
reference direction R@(0,1,0); the arrow drawn in the local tangent plane
shows that direction, so the in-plane angle encodes the psi twist.

Colour runs base->tip via Paul Tol's colour-blind-safe 'sunset' scheme.
"""

# Paul Tol 'sunset' (colour-blind safe), base -> tip.
_TOL_SUNSET = [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF",
    "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026",
]
_TOL_SUNSET_SCALE = [
    [i / (len(_TOL_SUNSET) - 1), c] for i, c in enumerate(_TOL_SUNSET)
]

# Paul Tol high-contrast qualitative scheme
_TOL_HC_BLUE   = "#004488"
_TOL_HC_YELLOW = "#DDAA33"
_TOL_HC_RED    = "#BB5566"

# 2-stop gradient: blue (arrow back) -> gold (arrow tip)
_TOL_HC_SCALE = [[0.0, _TOL_HC_BLUE], [1.0, _TOL_HC_YELLOW]]


def _unit_sphere(n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def plot_SO3_orientations(
    Rs,
    plot_psi_as_arrow: bool = True,
    highlight_indices=None,
    arrow_size: float = 0.12,
    sphere_opacity: float = 0.12,
    arrows_in_sunset_colorscale: bool = False,
    highlight_as_dot_halo: bool = False,
    title: str = "",
):
    """Plot a distribution of SO(3) orientations on the unit sphere.

    Parameters
    ----------
    Rs : (n, 3, 3) array of rotation matrices.
    plot_psi_as_arrow : bool
        True -> draw an oriented triangle arrow per orientation (shows psi).
        False -> draw a dot per orientation (fast for large n; no psi).
    highlight_indices : sequence of int or None
        Orientations to ring with a green contour (arrow outline or dot halo).
    arrow_size : float
        Arrow length in sphere radii.
    sphere_opacity : float
        Opacity of the reference unit sphere.
    """
    Rs = np.asarray(Rs)
    n = Rs.shape[0]
    hi = set() if highlight_indices is None else set(int(i) for i in highlight_indices)

    # base points and reference directions for all orientations at once
    base = Rs @ np.array([0.0, 0.0, 1.0])      # (n, 3) point on sphere
    refdir = Rs @ np.array([1.0, 0.0, 0.0])    # (n, 3) arrow direction (pointing along x direction)

    traces = []

    # --- the reference unit sphere ---
    sx, sy, sz = _unit_sphere()
    traces.append(
        go.Surface(
            x=sx, y=sy, z=sz, opacity=sphere_opacity,
            colorscale=[[0, "#cfd4da"], [1, "#cfd4da"]],
            showscale=False, hoverinfo="skip", name="unit sphere",
        )
    )

    if plot_psi_as_arrow:
        # project the reference direction into the tangent plane at base,
        # so the arrow lies flat on the sphere surface.
        radial = base                                       # outward normal
        tang = refdir - (np.sum(refdir * radial, axis=1, keepdims=True)) * radial
        norm = np.linalg.norm(tang, axis=1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)            # guard degenerate
        tang = tang / norm                                  # unit tangent (arrow dir)
        # a sideways vector in the tangent plane for triangle width
        side = np.cross(radial, tang)

        L = arrow_size
        Wt = arrow_size * 0.45                              # half-width at base
        tip = base + L * tang
        bl = base - 0.0 * tang + Wt * side                 # base-left
        br = base - 0.0 * tang - Wt * side                 # base-right

        # one Mesh3d per arrow with per-vertex intensity (base=0, tip=1)
        for i in range(n):
            verts = np.array([bl[i], br[i], tip[i]])        # (3,3)
            traces.append(
                go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=[0], j=[1], k=[2],
                    intensity=[0.0, 0.0, 1.0],              
                    colorscale=_TOL_HC_SCALE if not arrows_in_sunset_colorscale else _TOL_SUNSET_SCALE,       
                    #colorscale=_TOL_HC_SCALE,       # back blue, tip gold
                    #colorscale=_TOL_SUNSET_SCALE,   # back blue, tip red
                    cmin=0.0, cmax=1.0,
                    showscale=False, hoverinfo="skip",
                    flatshading=True,
                )
            )
            # # highlight: green outline around the triangle
            # if i in hi:
            #     loop = np.array([bl[i], br[i], tip[i], bl[i]])
            #     traces.append(
            #         go.Scatter3d(
            #             x=loop[:, 0], y=loop[:, 1], z=loop[:, 2],
            #             mode="lines", 
            #             #line=dict(color="#1a9850", width=5),
            #             line=dict(color=_TOL_HC_RED, width=5),    # arrow outline
            #             hoverinfo="skip", showlegend=False,
            #         )
            #     )
    else:
        # dot mode: a single fast Scatter3d for all points
        traces.append(
            go.Scatter3d(
                x=base[:, 0], y=base[:, 1], z=base[:, 2],
                mode="markers",
                marker=dict(size=3, 
                            color=_TOL_HC_BLUE,
                            #color=np.arange(n),
                            #colorscale=_TOL_SUNSET_SCALE, 
                            showscale=False),
                hoverinfo="skip", showlegend=False,
            )
        )
    if hi:
        marker_red_dot = dict(size=7, color=_TOL_HC_RED)
        marker_red_circle = dict(size=7, color="rgba(0,0,0,0)", line=dict(color=_TOL_HC_RED, width=4000))   # dot halo
        idx = np.array(sorted(hi))
        traces.append(
            go.Scatter3d(
                x=base[idx, 0], y=base[idx, 1], z=base[idx, 2],
                mode="markers",
                #marker=dict(size=8, color="rgba(0,0,0,0)",
                #            line=dict(color="#1a9850", width=4)),
                marker=marker_red_dot if highlight_as_dot_halo else marker_red_circle,
                hoverinfo="skip", showlegend=False,
            )
        )

    #lines = [title + f"<b>{mol.name}</b> ({mol.force_field})"]
    #title = "<br>".join(lines)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], title="x"),
            yaxis=dict(range=[-1.2, 1.2], title="y"),
            zaxis=dict(range=[-1.2, 1.2], title="z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"SO(3) orientation samples (n = {n})",
            subtitle=dict(text=title) if title else None,
        ),
        #title=f"SO(3) orientation samples (n = {n}){f"<br>{title}" if title else ""}",
        #title=dict(text=title, x=0.02, font=dict(size=12)),
        showlegend=False,
    )
    return fig

# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, ".")
#     #from orientations import orientation_matrices

#     #Rs = np.asarray(orientation_matrices(300))
#     fig = plot_SO3_orientations(Rs, plot_psi_as_arrow=True, arrows_in_sunset_colorscale=True,
#                                 highlight_indices=[0, 50, 150])
#     #fig.write_html("so3_arrows.html")
#     #print("wrote so3_arrows.html")
#     fig.show()

#     fig2 = plot_SO3_orientations(Rs, plot_psi_as_arrow=False,
#                                  highlight_indices=[0, 50, 150])
#     #fig2.write_html("so3_dots.html")
#     #print("wrote so3_dots.html")
#     fig2.show()


if __name__ == "__main__":
    import io, yaml
    from pathlib import Path
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
    #mol = Molecule.from_yaml("/Users/bene/Library/Mobile Documents/com~apple~CloudDocs/documents/eth/PhD/research/V_ext/input_data/molecule_ff/CO2_TraPPE_test.yaml")
    fig = visualize_molecule(mol)
    fig.write_html("nh3_view.html")
    print("wrote nh3_view.html")

    # unity rotation (no rotation at all)
    test = np.array([[ 1.0, 0.0, 0.0],
                  [ 0.0, 1.0, 0.0],
                  [ 0.0, 0.0, 1.0]],)
    # plot only one arrow at its original position (z=1, x=y=0) pointing in the x direction (psi=0)
    fig = plot_SO3_orientations(Rs=np.array([test]), plot_psi_as_arrow=True, arrows_in_sunset_colorscale=True,
                                    highlight_indices=[])
    fig.show()