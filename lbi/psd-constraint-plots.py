"""Code to plot the PSDs inferred from LBI fits to the floored data errors.

Author: Harry Bevins
Date: October 2025
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM
from fgivenx import plot_contours
from matplotlib.patches import FancyArrowPatch

from burstiness.prior import prior as prior_dict
from burstiness.utils import calc_extreg_psd, calc_sho_psd

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

plt.rcParams.update({"font.size": 8})

kernel = sys.argv[1]  # "ext-reg" or "sho"
nsamples = 10000
nparams = 2500
ndims = 5 if kernel == "ext-reg" else 3
nlive = ndims * 250

calc_psd = calc_sho_psd if kernel == "sho" else calc_extreg_psd

if kernel == "sho":
    pnames = ["s0", "tau0", "Q"]
elif kernel == "ext-reg":
    pnames = ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]

log10_prior_lo = np.array(
    [np.log10(prior_dict[p]["min"]) for p in pnames]
)
log10_prior_hi = np.array(
    [np.log10(prior_dict[p]["max"]) for p in pnames]
)

with open(
    "observations/intrinsic_scatter_per_timescale_errors.yaml",
) as f:
    data = yaml.safe_load(f)

tau = np.linspace(1e-3, 5, 500)

redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
latex_m = [
    r"$8 \leq \log M_* / M_\odot \leq 8.5$",
    r"$8.5 < \log M_* / M_\odot \leq 9$",
    r"$9 < \log M_* / M_\odot \leq 9.5$",
    r"$9.5 < \log M_* / M_\odot \leq 10$",
]
mass_bins = {
    "M_A": "8.0_8.5", "M_B": "8.5_9.0",
    "M_C": "9.0_9.5", "M_D": "9.5_10.0",
}
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}

rng = np.random.default_rng(0)
prior_samples = 10 ** (
    rng.uniform(size=(250, ndims))
    * (log10_prior_hi - log10_prior_lo)
    + log10_prior_lo
)

for j, (z_val, z) in enumerate(zip(redshifts, red)):
    if z == "z6_7":
        naxis = 3
        mass_iterator = list(enumerate(masses[:-1]))
        fig_size = 4.725
    elif z == "z7_8":
        naxis = 2
        mass_iterator = list(enumerate(masses[:-2]))
        fig_size = 3.15
    else:
        naxis = 4
        fig_size = 6.3
        mass_iterator = list(enumerate(masses))

    fig, ax = plt.subplots(
        1, naxis, figsize=(fig_size, 3), sharex=True, sharey=True
    )

    for i, mass in mass_iterator:
        target = f"{mass}_{z}"
        if target in skips:
            continue

        mb = mass_bins[mass]
        chains_dir = (
            f"lbi/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
            f"-{nparams}-{nsamples}_z_{z_val}_mass_bin_{mb}_nlive_{nlive}"
        )
        path = f"{chains_dir}/{target}/posterior.npz"

        try:
            theta = 10 ** np.load(path)["samples_log10"]  # physical space
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue

        mean_psd = np.median(
            [calc_psd(tau, th) for th in theta], axis=0
        )

        plot_contours(
            calc_psd, tau, theta,
            ax=ax[i], alpha=0.5,
            contour_line_levels=[1, 2], fineness=1,
        )
        for th in prior_samples:
            ax[i].plot(tau, calc_psd(tau, th), alpha=0.1, c="gray", zorder=-10)

        ax[i].plot(
            tau, mean_psd, color="black", lw=2, label="Median Inferred PSD",
        )
        ax[i].set_xlabel(r"Timescale [Gyr]")
        ax[i].set_title(latex_m[i], fontsize=8)

    ax[0].set_ylabel(r"PSD [dex$^2$ Gyr]")

    for i in range(naxis):
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_ylim(bottom=1e-5)
        ax[i].axvline(0.1, ls="--", color="k")
        ax[i].add_patch(FancyArrowPatch(
            (1e-1, 1), (1e-2, 1),
            arrowstyle="-|>", mutation_scale=20, color="black",
        ))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    plt.savefig(
        f"lbi/{kernel}_psd_redshift_{z}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
