import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Every function draws on `ax` if one is given, otherwise creates its own,
# and returns the Axes.


def plot_kymograph(history, x, ax=None):
    """
    Space-time plot of a 1D field: position on the horizontal axis,
    snapshot index on the vertical, concentration as colour.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        history,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], 0, len(history) - 1],
    )
    ax.figure.colorbar(im, ax=ax, label="Concentration")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Snapshot")
    ax.set_title("Kymograph")

    return ax


def plot_profile(u, v, x, ax=None):
    """
    Final 1D profiles of u and v against position on a shared axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(x, u, c="red", label="$u$")
    ax.plot(x, v, c="blue", label="$v$")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Concentration")
    ax.set_title("Final profile")
    ax.legend()

    return ax


def plot_pattern_2d(field, L=1.0, ax=None):
    """
    A single 2D concentration field on a square domain of side L,
    with a colourbar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        field,
        cmap="viridis",
        origin="lower",
        aspect="equal",
        extent=[0, L, 0, L],
    )
    ax.figure.colorbar(im, ax=ax, label="Concentration $u$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("2D Turing pattern")

    return ax


def plot_dispersion(k, growth, k_modes=None, growth_modes=None, ax=None):
    """
    Growth rate λ(k) as a continuous curve, with a line at λ = 0.
    If k_modes and growth_modes are given, mark the admissible discrete
    modes and highlight the fastest-growing one.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(k, growth, c="black", ls="--", alpha=0.6, label="Continuous dispersion")

    if k_modes is not None and growth_modes is not None:
        idx = np.argmax(growth_modes)
        n = idx + 1
        ax.scatter(k_modes, growth_modes, s=50, c="red", zorder=2,
                   label="Admissible modes $k_n$")
        ax.scatter([k_modes[idx]], [growth_modes[idx]], s=100, c="blue", zorder=3,
                   label=f"Predicted winner (n = {n}, peaks ≈ {n / 2:.1f})")

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel(r"Growth rate $\lambda(k)$")
    ax.set_title("Dispersion relation")
    ax.legend()

    return ax


def plot_threshold_slice(b_vals, Dc_vals, a, D_sim, runs=None, ax=None):
    """
    D_c as a function of b at fixed a, with the region above the curve
    shaded as predicted patterning and a line at the simulated D.

    runs: optional list of (b, patterned) pairs, drawn as filled markers
    if patterned is True and hollow if False.
    """
    b_vals = np.asarray(b_vals, dtype=float)
    Dc_vals = np.asarray(Dc_vals, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    finite = np.isfinite(Dc_vals)

    ax.plot(b_vals, Dc_vals, c="red", label="$D_c(b)$")
    ax.axhline(D_sim, c="black", ls="--", alpha=0.5,
               label=f"$D$ = {D_sim} (simulations)")

    if finite.any():
        y_top = 1.1 * max(np.max(Dc_vals[finite]), D_sim)
        ax.fill_between(b_vals, Dc_vals, y_top, color="red", alpha=0.2,
                        label="Pattern predicted")
        ax.axvline(b_vals[finite][0], c="red", ls="--", alpha=0.6)
        ax.set_ylim(top=y_top)

    if runs is not None:
        patterned = [b for b, ok in runs if ok]
        decayed = [b for b, ok in runs if not ok]
        if patterned:
            ax.scatter(patterned, np.full(len(patterned), D_sim),
                       marker="^", color="black", zorder=3, label="Patterned")
        if decayed:
            ax.scatter(decayed, np.full(len(decayed), D_sim),
                       marker="^", edgecolors="black", facecolors="none",
                       zorder=3, label="Decayed")

    ax.set_xlabel("Parameter $b$")
    ax.set_ylabel("Critical diffusion ratio $D_c$")
    ax.set_title(f"Parameter $a$ fixed at {a}")
    ax.legend(loc="lower right")

    return ax


def plot_threshold_map(a_vals, b_vals, Dc_grid, D_sim, runs=None, ax=None,vmax=1e3):
    """
    Heatmap of D_c over the (a, b) plane on a log colour scale, with the
    D = D_sim contour and the line b = a. NaN regions are left blank.

    runs: optional list of (a, b, patterned) triples, drawn as filled
    markers if patterned is True and hollow if False.
    """
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    Dc_grid = np.asarray(Dc_grid, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    finite = Dc_grid[np.isfinite(Dc_grid)]
    extent = [a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]]

    if finite.size:
        norm = LogNorm(vmin=finite.min(), vmax=vmax)
        im = ax.imshow(Dc_grid, cmap="viridis", origin="lower",
                       aspect="auto", extent=extent, norm=norm)
        ax.figure.colorbar(im, ax=ax, label="Critical diffusion ratio $D_c$ (log scale)")

        if finite.min() < D_sim < finite.max():
            c = ax.contour(a_vals, b_vals, Dc_grid, levels=[D_sim],
                           colors="white", linewidths=1.5)
            ax.clabel(c, fmt={D_sim: f"D = {D_sim}"}, inline=True, fontsize=10)

    ax.plot(a_vals, a_vals, ls="--", c="red", label="$b = a$")

    if runs is not None:
        patterned = np.array([(ra, rb) for ra, rb, ok in runs if ok])
        decayed = np.array([(ra, rb) for ra, rb, ok in runs if not ok])
        if len(patterned):
            ax.scatter(patterned[:, 0], patterned[:, 1], marker="^",
                       color="black", zorder=3, label="Patterned")
        if len(decayed):
            ax.scatter(decayed[:, 0], decayed[:, 1], marker="^",
                       edgecolors="black", facecolors="none", zorder=3,
                       label="Decayed")

    ax.set_xlim(a_vals[0], a_vals[-1])
    ax.set_ylim(b_vals[0], b_vals[-1])
    ax.set_xlabel("Parameter $a$")
    ax.set_ylabel("Parameter $b$")
    ax.set_title("Critical diffusion ratio $D_c(a, b)$")
    ax.legend(loc="upper left")

    return ax