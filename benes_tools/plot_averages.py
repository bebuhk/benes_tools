from .cDFT_ext_pot import canonical_average, FEA_Abraham_ns
import matplotlib.pyplot as plt
import numpy as np

def plot_canonical_average_analysis(E_sum_K_na, temperature_K = 298.15, return_Ew=False):
    """Plot the distributions of total energies across orientations for a grid point.
    Input:
    - E_sum_K_na: (N_orientations,) array of total energies in K
    - temperature_K: temperature in K for Boltzmann weighting
    - return_Ew: if True, return the Boltzmann-weighted mean energy Ew
    """
    # Paul Tol some colors
    _TOL_HC_YELLOW = "#DDAA33"
    _TOL_HC_GREEN  = "#228833"

    Ew, w = canonical_average(E_sum_K_na, temperature_K)

    # Two plots: full dual-axis plot, and a second energy-only plot with y-cutoff at 1000 K
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)


    x = np.arange(E_sum_K_na.shape[0])

    # Left axis: energy
    ax1.plot(x, E_sum_K_na, label='Total Energy', color=_TOL_HC_GREEN)
    ax1.axhline(np.average(E_sum_K_na), color=_TOL_HC_GREEN, linestyle='--',
            label=f'Average Total = {np.average(E_sum_K_na):.2f} K')
    ax1.axhline(0, color='gray', linestyle='dashed')
    ax1.set_xlabel('angle index')
    ax1.set_ylabel('Energy (K)', color=_TOL_HC_GREEN)
    ax1.tick_params(axis='y', labelcolor=_TOL_HC_GREEN)

    # Right axis: weights
    ax2 = ax1.twinx()
    ax2.plot(x, w, label='Boltzmann weight', color=_TOL_HC_YELLOW)
    ax2.set_ylabel('Weight', color=_TOL_HC_YELLOW)
    ax2.tick_params(axis='y', labelcolor=_TOL_HC_YELLOW)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax1.set_title(f'Energy and Boltzmann weight vs angle index (sum of weights = {np.sum(w):.2e})')

    # Bottom plot: energy only, capped at 1000 K
    ax3.plot(x, E_sum_K_na, color=_TOL_HC_GREEN, label='Total Energy. (min={:.2f} K)'.format(np.min(E_sum_K_na)))
    ax3.axhline(np.average(E_sum_K_na), color=_TOL_HC_GREEN, linestyle='--',
            label=f'Average Total = {np.average(E_sum_K_na):.2f} K')
    ax3.axhline(Ew, color='black', linestyle='dashed', label=f'Boltzmann-weighted Mean = {Ew:.2f} K = {Ew/temperature_K:.6f} (ul)')
    ax3.axhline(0, color='gray', linestyle='dashed')
    ax3.set_xlabel('angle index')
    ax3.set_ylabel('Energy (K)')
    ax3.set_ylim(top=1000)
    ax3.legend(loc='best')
    ax3.set_title(f'Total energy vs angle index (cutoff at 1000 K) - Boltzmann-weighted mean = {Ew:.2f} K = {Ew/temperature_K:.6f} (ul)')

    plt.tight_layout()
    plt.show()

    if return_Ew:
        return Ew


def plot_canonical_and_FEA_average_analysis(E_sum_K_na, temperature_K = 298.15, return_Ew=False, title="", y_lim_K=1000, save_path=None, show_plot=True, skip_ns_comparison=False):
    """Plot the distributions of total energies across orientations for a grid point.
    Input:
    - E_sum_K_na: (N_orientations,) array of total energies in K
    - temperature_K: temperature in K for Boltzmann weighting
    - return_Ew: if True, return the Boltzmann-weighted mean energy Ew
"""
    # Paul Tol some colors
    _TOL_HC_GREEN  = "#228833"
        # Paul Tol high-contrast qualitative scheme
    _TOL_HC_BLUE   = "#004488"
    _TOL_HC_YELLOW = "#DDAA33"
    _TOL_HC_RED    = "#BB5566"

    Ew, w = canonical_average(E_sum_K_na, temperature_K)

    e2E = np.exp(-E_sum_K_na / temperature_K)
    Ew_Abraham = -temperature_K * np.log((np.sum(e2E, axis=0))/len(E_sum_K_na))

    Ew_FEA = FEA_Abraham_ns(E_sum_K_na, temperature_K)
    if not skip_ns_comparison:
        assert np.isclose(Ew_Abraham, Ew_FEA), "FEA Abraham Ew (with numerically stable function) does not match local computation..."
    elif not np.isclose(Ew_Abraham, Ew_FEA):
        print(f"Warning: FEA Abraham Ew (with numerically stable function) does not match local computation: {Ew_Abraham} vs {Ew_FEA} K")
    assert np.allclose(w, e2E), "Boltzmann weights do not match expected exp(-E/kT) values..."
        
    # Two plots: full dual-axis plot, and a second energy-only plot with y-cutoff at 1000 K
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if title:
        plt.suptitle(title, fontsize=12)


    x = np.arange(E_sum_K_na.shape[0])

    # Left axis: energy
    ax1.plot(x, E_sum_K_na, label='Total Energy E', color=_TOL_HC_GREEN)
    ax1.axhline(np.average(E_sum_K_na), color=_TOL_HC_GREEN, linestyle='--',
            label=f'Average Total = {np.average(E_sum_K_na):.2f} K')
    ax1.axhline(0, color='gray', linestyle='dashed')
    ax1.set_xlabel('angle index (total orientations = {})'.format(E_sum_K_na.shape[0]))
    ax1.set_ylabel('Energy (K)', color=_TOL_HC_GREEN)
    ax1.tick_params(axis='y', labelcolor=_TOL_HC_GREEN)

    # Right axis: weights
    ax2 = ax1.twinx()
    ax2.plot(x, w, label='Boltzmann weight (= exp(-E/kT))', color=_TOL_HC_YELLOW)
    ax2.set_ylabel('Weight', color=_TOL_HC_YELLOW)
    ax2.tick_params(axis='y', labelcolor=_TOL_HC_YELLOW)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax1.set_title(f'Energy and Boltzmann weight vs angle index (sum of weights = {np.sum(w):.2e})')

    # Bottom plot: energy only, capped at 1000 K
    ax3.plot(x, E_sum_K_na, color=_TOL_HC_GREEN, label='Total Energy E. (min={:.2f} K)'.format(np.min(E_sum_K_na)))
    ax3.axhline(np.average(E_sum_K_na), color=_TOL_HC_GREEN, linestyle='--',
            label=f'Average Total = {np.average(E_sum_K_na):.2f} K')
    ax3.axhline(Ew, color=_TOL_HC_BLUE, linestyle='dashed', label=f'canonical average = {Ew:.2f} K = {Ew/temperature_K:.6f} (ul)')
    ax3.axhline(Ew_FEA, linestyle='dashed', label=f'FEA = {Ew_FEA:.2f} K = {Ew_FEA/temperature_K:.6f} (ul) = -T*ln[sum(weights)/N]', color=_TOL_HC_RED)
    ax3.axhline(0, color='gray', linestyle='dashed')
    ax3.set_xlabel('angle index')
    ax3.set_ylabel('Energy (K)')
    ax3.set_ylim(top=y_lim_K)
    ax3.legend(loc='best')
    ax3.set_title(f'Total energy vs angle index (cutoff at {y_lim_K} K) - Boltzmann-weighted mean = {Ew:.2f} K = {Ew/temperature_K:.6f} (ul)')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_Ew:
        return Ew


## bene 12.06.2026: this is not an bolzman nor FEA average but the LJ and Coulomb part that constitute the total energy.
def plot_energy_distributions(energies_LJ_K_na, energies_C_K_na, E_sum_K_na=None, title_suffix='', save_path=None,
                              label_LJ='LJ', label_C='Coulomb', label_Total='Total', show_plot=True):
    """Plot the distributions of LJ, Coulomb, and total energies across orientations for one grid point.
    Input:
    - energies_LJ_K_na: (N_orientations, N_grid_points) array of LJ energies in K.
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

    LJ_array = energies_LJ_K_na # with len = n_orientations 
    LJ_avg = np.mean(LJ_array)
    LJ_min = np.min(LJ_array)
    LJ_max = np.max(LJ_array)
    C_array = energies_C_K_na # with len = n_orientations 
    C_avg = np.mean(C_array)
    C_min = np.min(C_array)
    C_max = np.max(C_array)
    Total_array = E_sum_K_na if E_sum_K_na is not None else LJ_array + C_array # with len = n_orientations
    Total_avg = np.mean(Total_array)
    Total_min = np.min(Total_array)
    Total_max = np.max(Total_array)
    n = len(LJ_array)
    assert len(C_array) == n and len(Total_array) == n, "Input arrays must have the same length."

    # plot energies over angle index
    plt.figure(figsize=(10, 4))
    plt.suptitle(f'Energy distributions across orientations\n{title_suffix}', fontsize=10)
    plt.subplot(1, 2, 1)
    plt.plot(LJ_array, label=label_LJ, color=_TOL_HC_RED)
    plt.hlines(LJ_avg, 0, energies_LJ_K_na.shape[0]-1, colors=_TOL_HC_RED, linestyles='dashed', label='Average {} = {:.2f} K'.format(label_LJ.split("(")[0], LJ_avg))
    plt.plot(C_array, label=label_C, color=_TOL_HC_BLUE)
    plt.hlines(C_avg, 0, energies_LJ_K_na.shape[0]-1, colors=_TOL_HC_BLUE, linestyles='dashed', label='Average {} = {:.2f} K'.format(label_C.split("(")[0], C_avg))
    plt.xlabel('angle index')
    plt.ylabel('Energy (K)')
    plt.title('Energy vs angle index\nmin-max {:.2f}..{:.2f} K (LJ), {:.2f}..{:.2f} K (Coul)'.format(LJ_min, LJ_max, C_min, C_max), fontsize=10)
    plt.legend(fontsize=8)
    plt.subplot(1, 2, 2)
    plt.plot(LJ_array, label=label_LJ, color=_TOL_HC_RED)
    plt.plot(C_array, label=label_C, color=_TOL_HC_BLUE)
    plt.hlines(Total_avg, 0, energies_LJ_K_na.shape[0]-1, colors=_TOL_HC_YELLOW, linestyles='dashed', label='Average {} = {:.2f} K'.format(label_Total.split("(")[0], Total_avg))
    plt.plot(Total_array, label=label_Total, color=_TOL_HC_YELLOW)
    plt.xlabel('angle index')
    plt.ylabel('Energy (K)')
    plt.title('Total energy vs angle index\n min-max {:.2f}..{:.2f} K'.format(Total_min, Total_max), fontsize=10)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    pass