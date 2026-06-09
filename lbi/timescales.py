"""Violin plots of parameter posteriors vs stellar mass and redshift per kernel."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def tau_merge_dekel(log_Mstar, z):
    """Merger timescale from Dekel+2020, collapsed form (eq. 15 rearranged).

    https://arxiv.org/pdf/1912.08213
    
    So we start with eq 5. 

    tmer = 1.1Gyr *f_{sv,-2.5} *f_{sb,0.5}^{-1} * \
            tau_{30} * f_{b, 0.16}^{-1} * beta * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}
        
    Then we say tau_{30}, beta, f_{b, 0.16} = 1 so

    tmer = 1.1Gyr *f_{sv,-2.5} *f_{sb,0.5}^{-1} * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}

    Then we say f_{sv} approx 3e-3*f_{sv, -2.5} * M_{v, 11}
    and f_{sv, -2.5}
    So we start with eq 5. 

    tmer = 1.1Gyr *f_{sv,-2.5} *f_{sb,0.5}^{-1} * \
            tau_{30} * f_{b, 0.16}^{-1} * beta * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}
        
    Then we say tau_{30}, beta, f_{b, 0.16} = 1 so

    tmer = 1.1Gyr *f_{sv,-2.5} *f_{sb,0.5}^{-1} * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}

    f_{sv, -2.5} = (M_s / M_v) / 10^{-2.5}
    f_{sb, 0.5}^{-1} = f_{sb}^{-1} / 0.5^{-1} = (M_b / M_s)/2

    tmer = 1.1Gyr * (M_s / M_v) / 10^{-2.5} * (M_b / M_s)/2 * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}

    tmer = 1.1Gyr / (2e-2.5) * (M_b / M_v) * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}

    Use abundance matching
    Eq. 10 M_s / M_v = 3e-3 * M_{v, 11} so M_s proportional to M_v^2
    Eq. 11 M_b / M_s = f_{sb}^{-1} = 1/1.27 * (1 + z) * M_{s, 11}^{-0.14} \
        = 1/1.27 * (1 + z) * (3e-3 * M_{v, 11})^{-0.28}

    So M_b/M_v = M_b/M_s * M_s/M_v = 1/1.27 * (1 + z) * (3e-3 * M_{v, 11})^{-0.28} * 3e-3 * M_{v, 11}
        = 1/1.27 * (1 + z) * (3e-3)^{0.72} * M_{v, 11}^{0.72}

    tmer = 1.1Gyr / (2e-2.5) * 1/1.27 * (1 + z) * (3e-3)^{0.72} * M_{v, 11}^{0.72} * \
            (1 + z)^{-5/2} * M_{v, 11}^{-0.1}
        = 1.1Gyr / (2e-2.5) * 1/1.27 * (3e-3)^{0.72} * (1 + z)^{-3/2} * M_{v, 11}^{0.62}

    tmer = 0.5 * M_{v, 11}^{0.62} * (1 + z)^{-3/2} Gyr    

    Parameters
    ----------
    log_Mstar : float or array
        log10(M*/Msun)
    z : float or array
        redshift
    
    Returns
    -------
    t_mer in Gyr
    """
    Mv_11 = 10**log_Mstar / (3e-3 * 1e11)
    return 0.5 * Mv_11**0.62 * (1 + z) ** (-1.5)  # Gyr

def tau_merge(log_Mstar, z):
    """From Puskas et al. 2025."""
    if log_Mstar < 8.5:
        return 1/(2.20e-2 *(1 + z)**(4.18) * np.exp(-0.42 * (1 +z)))
    elif log_Mstar < 9.0:
        return 1/(2.25e-2 *(1 + z)**(3.36) * np.exp(-0.27 * (1 +z)))
    elif log_Mstar <= 10:
        return 1/(2.61e-2 *(1 + z)**(2.90) * np.exp(-0.2 * (1 +z)))


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

plt.rcParams.update({"font.size": 8})

nsamples = 10000
nparams = 2500
nlive = {"sho": 3 * 250, "ext-reg": 5 * 250}

redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
mass_values = [8.25, 8.75, 9.25, 9.75]
masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
red_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]
mass_labels = [
    r"A: $[8, 8.5]$",
    r"B: $(8.5, 9]$",
    r"C: $(9, 9.5]$",
    r"D: $(9.5, 10]$",
]
mass_bins = {
    "M_A": "8.0_8.5",
    "M_B": "8.5_9.0",
    "M_C": "9.0_9.5",
    "M_D": "9.5_10.0",
}
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}

param_info = {
    "sho": {
        "names": ["s0", "tau0", "Q"],
        "labels": [
            r"$\log S_0$",
            r"$\log \tau_0\ [{\rm Gyr}]$",
            r"$\log Q$",
        ],
    },
    "ext-reg": {
        "names": ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"],
        "labels": [
            r"$\log \sigma_{\rm Reg}$",
            r"$\log \tau_{\rm eq}\ [{\rm Gyr}]$",
            r"$\log \tau_{\rm in}\ [{\rm Gyr}]$",
            r"$\log \sigma_{\rm Dyn}$",
            r"$\log \tau_{\rm Dyn}\ [{\rm Gyr}]$",
        ],
    },
}

z_colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(redshifts)))

# Violin geometry: within each mass group, redshift violins are spaced by
# z_spacing; groups are separated by an additional mass_gap.
violin_width = 0.3
z_spacing = 0.17
mass_gap = 0.45

os.makedirs("lbi/figs", exist_ok=True)

for kernel in ["sho", "ext-reg"]:
    pnames = param_info[kernel]["names"]
    plabels = param_info[kernel]["labels"]
    ndims = len(pnames)

    chain_data = {}
    for j, (z, ztag) in enumerate(zip(redshifts, red)):
        for i, mass in enumerate(masses):
            target = f"{mass}_{ztag}"
            if target in skips:
                continue
            mb = mass_bins[mass]
            folder = (
                f"lbi/chains/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
                f"-{nparams}-{nsamples}_z_{z}_mass_bin_{mb}"
                f"_nlive_{nlive[kernel]}"
            )
            npz_path = f"{folder}/{target}/posterior.npz"
            if not os.path.exists(npz_path):
                continue
            chain_data[(i, j)] = np.load(npz_path)["samples_log10"]

    if not chain_data:
        print(f"No chains found for {kernel}, skipping.")
        continue

    timescale_redshifts = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    nrows, ncols = (2, 3) if ndims == 5 else (1, ndims)
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(6.3, 3.0 * nrows),
        squeeze=False,
    )
    axes_flat = axes_grid.flatten()

    if ndims == 5:
        axes_flat[5].set_visible(False)

    n_z = len(redshifts)
    group_span = (n_z - 1) * z_spacing
    group_pitch = group_span + mass_gap
    group_centers = [i * group_pitch for i in range(len(masses))]

    for p in range(ndims):
        ax = axes_flat[p]
        for i in range(len(masses)):
            for j in range(len(red)):
                if (i, j) not in chain_data:
                    continue
                samples = chain_data[(i, j)][:, p]
                x_pos = group_centers[i] + (j - (n_z - 1) / 2) * z_spacing

                vp = ax.violinplot(
                    samples,
                    positions=[x_pos],
                    widths=violin_width,
                    showmedians=True,
                    showextrema=False,
                )
                for body in vp["bodies"]:
                    body.set_facecolor(z_colors[j])
                    body.set_alpha(0.75)
                    body.set_edgecolor("none")
                vp["cmedians"].set_color("black")
                vp["cmedians"].set_linewidth(1.2)

        ax.set_xticks(group_centers)
        ax.set_xticklabels(mass_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(plabels[p], fontsize=8)
        ax.set_xlabel(r"$\log M_* / M_\odot$", fontsize=8)
        ax.grid(axis="y", lw=0.5, alpha=0.4)

    legend_patches = [
        mpatches.Patch(facecolor=z_colors[j], label=rf"$z={red_labels[j]}$")
        for j in range(len(redshifts))
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(redshifts),
        fontsize=7,
        title_fontsize=7,
        frameon=False,
    )

    kernel_name = "SHO" if kernel == "sho" else "ExtReg"
    fig.suptitle(kernel_name, fontsize=9)
    plt.tight_layout()
    out = f"lbi/figs/parameter_violin_{kernel}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    nrows, ncols = (1, len(mass_bins))
    fig, axes_flat = plt.subplots(
        nrows,
        ncols,
        figsize=(8, 4.0 * nrows),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    times = ['tau0'] if kernel == 'sho' else ['tau_dyn'] 
    for i in range(len(masses)):
        ax = axes_flat[0, i]
        dynamical_times = []
        halo_dynamical_times = []
        merger_times = []
        hubble_times_0_1 = []
        tages = []
        plot_redshifts = []
        for j in range(len(timescale_redshifts)):
            tage = cosmo.age(timescale_redshifts[j]).value
            tdyn = (
                100
                * (10 ** mass_values[i] / 10**10) ** (0.2)
                * ((1 + timescale_redshifts[j]) / 3) ** (-3 / 2)
                * 1e-3
            )
            tdyn_halo = 1.4*((1+timescale_redshifts[j])/3)**(-3/2)
            tmerge = tau_merge(mass_values[i], timescale_redshifts[j])
            dynamical_times.append(np.log10(tdyn))
            halo_dynamical_times.append(np.log10(tdyn_halo))
            merger_times.append(np.log10(tmerge))
            tages.append(np.log10(tage))
            hubble_times_0_1.append(np.log10(tage * 0.1))

        for j in range(len(red)):
            if (i, j) not in chain_data: 
                continue
            samples = chain_data[(i, j)][:, pnames.index(times[0])]

            vp = ax.violinplot(
                samples,
                positions=[redshifts[j]],
                widths=violin_width,
                showmedians=True,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(z_colors[j])
                body.set_alpha(0.75)
                body.set_edgecolor("none")
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.2)

            
        ax.fill_between(
            timescale_redshifts,
            tages,
            [0.5] * len(tages),
            color="gray",
            alpha=0.2,
        )

        ax.plot(
            timescale_redshifts,
            dynamical_times,
            label="Gal. Dynamical Time",
            color='grey',
            ls='-'
        )
        ax.plot(
            timescale_redshifts,
            halo_dynamical_times,
            label="Halo Dynamical Time",
            color='grey',
            ls='--'
        )
        ax.plot(
            timescale_redshifts,
            merger_times,
            label=r"Merger Time (Puk$\acute{a}$s+2025)",
            color='grey',
            ls=(0, (3, 1, 1, 1))
        )
        ax.plot(
            timescale_redshifts,
            hubble_times_0_1,
            color="gray",
            label="0.1 x Age of Universe",
            ls=':'
        )
        ax.set_ylim(-3, 0.5)
        ax.set_xlim(2.5, 8.5)
        ax.set_xticks(redshifts)
        ax.set_xticklabels(red_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(mass_labels[i], fontsize=8)
        ax.set_xlabel(r"Redshift", fontsize=8)
        ax.grid(axis="y", lw=0.5, alpha=0.4)
        if i == 0:
            ax.set_ylabel(param_info[kernel]['labels'][pnames.index(times[0])], fontsize=8)
    if kernel == "sho":
        axes_flat[0, 0].legend(loc="lower left", fontsize=6)
    #kernel_name = "SHO" if kernel == "sho" else "ExtReg"
    #fig.suptitle(kernel_name, fontsize=9)
    plt.tight_layout()
    out = f"lbi/figs/parameter_timescales_{kernel}_tau_merge.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
