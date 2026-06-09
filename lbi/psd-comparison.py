"""3-row x 4-col PSD comparison at z=3-4: SHO, ext-reg, and difference."""

import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours

from burstiness.prior import prior as prior_dict
from burstiness.utils import calc_extreg_psd, calc_sho_psd

plt.rcParams.update({"font.size": 8})

nsamples = 10000
nparams = 2500
nlive = {"sho": 3 * 250, "ext-reg": 5 * 250}

z_val = 3.5
z = "z3_4"
masses = ["M_A", "M_B", "M_C", "M_D"]
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

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
t_obs = cosmo.age(z_val).to("Gyr").value
t_form = cosmo.age(30).to("Gyr").value
t_evolve = t_obs - t_form  # total evolution time

tau = np.logspace(-3, np.log10(t_evolve), 500)

# Prior samples for background
rng = np.random.default_rng(0)
prior_samples = {}
for kernel, pnames in [
    ("sho", ["s0", "tau0", "Q"]),
    ("ext-reg", ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]),
]:
    lo = np.array([np.log10(prior_dict[p]["min"]) for p in pnames])
    hi = np.array([np.log10(prior_dict[p]["max"]) for p in pnames])
    prior_samples[kernel] = 10 ** (
        rng.uniform(size=(250, len(pnames))) * (hi - lo) + lo
    )

fig, axes = plt.subplots(
    3, 4, figsize=(6.3, 7),
    sharex=True, sharey='row',
)

calc_psd = {"sho": calc_sho_psd, "ext-reg": calc_extreg_psd}
row_labels = ["SHO", "ExtReg", r"$\log_{10}(\frac{{\rm ExtReg}}{{\rm SHO}})$"]

posterior = {}
for kernel in ["sho", "ext-reg"]:
    posterior[kernel] = {}
    for mass in masses:
        target = f"{mass}_{z}"
        folder = (
            f"lbi/chains/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
            f"-{nparams}-{nsamples}_z_{z_val}_mass_bin_{mass_bins[mass]}"
            f"_nlive_{nlive[kernel]}"
        )
        path = f"{folder}/{target}/posterior.npz"
        try:
            posterior[kernel][mass] = 10 ** np.load(path)["samples_log10"]
        except FileNotFoundError:
            print(f"File not found: {path}")
            posterior[kernel][mass] = None

# Row 0: SHO, Row 1: ext-reg
for row, kernel in enumerate(["sho", "ext-reg"]):
    for i, mass in enumerate(masses):
        ax = axes[row, i]
        theta = posterior[kernel][mass]
        if theta is None:
            ax.axis("off")
            continue

        for th in prior_samples[kernel]:
            ax.plot(
                tau, calc_psd[kernel](tau, th),
                alpha=0.4, c="gray", lw=0.8, zorder=-10,
            )
        plot_contours(
            calc_psd[kernel], tau, theta,
            ax=ax, colors=plt.get_cmap(
                "Reds_r"
            ),
            contour_line_levels=[1, 2], fineness=1,
            alpha=0.8
        )
        median_psd = np.median(
            [calc_psd[kernel](tau, th) for th in theta], axis=0
        )
        ax.plot(tau, median_psd, color="black", lw=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)
        ax.axvline(0.01, ls=":", color='k')
        ax.axvline(0.1, ls=":", color='k')

# Row 2: difference using percentiles of randomly paired samples,
# consistent with fgivenx contour definition
for i, mass in enumerate(masses):
    ax = axes[2, i]
    theta_sho = posterior["sho"][mass]
    theta_er = posterior["ext-reg"][mass]
    if theta_sho is None or theta_er is None:
        ax.axis("off")
        continue

    n = min(len(theta_sho), len(theta_er))
    rng2 = np.random.default_rng(1)
    idx_sho = rng2.choice(len(theta_sho), n, replace=False)
    idx_er = rng2.choice(len(theta_er), n, replace=False)

    diffs = np.array([
         -np.log10(calc_sho_psd(tau, theta_sho[idx_sho[k]]))
          + np.log10(calc_extreg_psd(tau, theta_er[idx_er[k]]))
        for k in range(n)
    ]) 

    p2, p16, p50, p84, p98 = np.percentile(diffs, [2.5, 16, 50, 84, 97.5], axis=0)

    ax.plot(tau, p50, color="k", lw=1)
    cmap = plt.get_cmap("Reds_r")
    ax.fill_between(tau, p2, p98, color=cmap(0.7), label=r"$2\sigma$",
                    alpha=0.8)
    ax.fill_between(tau, p16, p84, color=cmap(0.2), label=r"$1\sigma$",
                    alpha=0.8)
    ax.axhline(0, color="k", ls="--")
    ax.axvline(0.01, ls=":", color="k")
    ax.axvline(0.1, ls=":", color="k")
    ax.set_xscale("log")

# Labels
for i, m_label in enumerate(latex_m):
    axes[0, i].set_title(m_label, fontsize=7)

for row, label in enumerate(row_labels):
    if row < 2:
        axes[row, 0].set_ylabel(
            label + "\n" + r"PSD [dex$^2$ Gyr]", fontsize=7
        )
    else:
        axes[row, 0].set_ylabel(
            label, fontsize=7
        )

for i in range(4):
    axes[2, i].set_xlabel(r"Timescale [Gyr]", fontsize=7)

for row in range(3):
    for i in range(4):
        axes[row, i].set_xticks([1e-2, 1e-1, 1e0])
        axes[row, i].set_xticklabels(
            [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"], fontsize=6
        )

# Legend for difference row
axes[2, 3].legend(fontsize=6, loc="upper right")

plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig("lbi/figs/psd_comparison_z3_4.png", dpi=300, bbox_inches="tight")
plt.close()
