"""Line plots of per-parameter KL divergence (posterior || prior) vs redshift."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from burstiness.prior import prior as prior_dict

plt.rcParams.update({"font.size": 8})

nsamples = 10000
nparams = 2500
nlive = {"sho": 750, "ext-reg": 1250}

param_info = {
    "sho": {
        "pnames": ["s0", "tau0", "Q"],
        "latex_names": [
            r"$\log S_0$",
            r"$\log \tau_0\,[{\rm Gyr}]$",
            r"$\log Q$",
        ],
    },
    "ext-reg": {
        "pnames": ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"],
        "latex_names": [
            r"$\log \sigma_{\rm Reg}$",
            r"$\log \tau_{\rm eq}\,[{\rm Gyr}]$",
            r"$\log \tau_{\rm in}\,[{\rm Gyr}]$",
            r"$\log \sigma_{\rm Dyn}$",
            r"$\log \tau_{\rm Dyn}\,[{\rm Gyr}]$",
        ],
    },
}

redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
mass_labels = [
    r"$8.0 \leq \log M_* / M_\odot < 8.5$",
    r"$8.5 \leq \log M_* / M_\odot < 9.0$",
    r"$9.0 \leq \log M_* / M_\odot < 9.5$",
    r"$9.5 \leq \log M_* / M_\odot < 10.0$",
]
mass_bins = {
    "M_A": "8.0_8.5",
    "M_B": "8.5_9.0",
    "M_C": "9.0_9.5",
    "M_D": "9.5_10.0",
}
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
markers = ["o", "s", "^", "D", "v"]

dkl_data = {}
for kernel, info in param_info.items():
    pnames = info["pnames"]
    ndims = len(pnames)
    log10_lo = np.array([np.log10(prior_dict[p]["min"]) for p in pnames])
    log10_hi = np.array([np.log10(prior_dict[p]["max"]) for p in pnames])
    log_prior_width = np.log(log10_hi - log10_lo)

    dkl = np.full((ndims, len(masses), len(redshifts)), np.nan)
    for i, mass in enumerate(masses):
        for j, (z_val, z) in enumerate(zip(redshifts, red)):
            if f"{mass}_{z}" in skips:
                continue
            folder = (
                f"lbi/chains/blackjax_chains_astroemu-{kernel}"
                f"-sfr_theory_mean-{nparams}-{nsamples}_z_{z_val}"
                f"_mass_bin_{mass_bins[mass]}_nlive_{nlive[kernel]}"
            )
            path = f"{folder}/{mass}_{z}/posterior.npz"
            try:
                samples = np.load(path)["samples_log10"]
            except FileNotFoundError:
                continue
            for p in range(ndims):
                s = samples[:, p]
                kde = gaussian_kde(s)
                dkl[p, i, j] = np.mean(np.log(kde(s)) + log_prior_width[p])
    dkl_data[kernel] = dkl

fig, axes = plt.subplots(
    2, 4, figsize=(7, 4.5), sharey=True, sharex=True
)

row_kernels = ["sho", "ext-reg"]
row_titles = ["SHO", "ExtReg"]

latex_z = [r"$3 < z \leq 4$", r"$4 < z \leq 5$", r"$5 < z \leq 6$", r"$6 < z \leq 7$", r"$7 < z \leq 8$"]

for row, kernel in enumerate(row_kernels):
    dkl = dkl_data[kernel]
    latex_names = param_info[kernel]["latex_names"]
    ndims = len(latex_names)

    for col, mass in enumerate(masses):
        ax = axes[row, col]
        for p in range(ndims):
            ax.plot(
                redshifts,
                dkl[p, col, :],
                color=colors[p],
                markersize=4,
                linewidth=1.2,
                label=latex_names[p],
            )
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_xticks(redshifts)
        ax.set_xticklabels(
            latex_z, fontsize=7,
            rotation=45,
        )
        if row == 0:
            ax.set_title(mass_labels[col], fontsize=7)
        if row == 1:
            ax.set_xlabel(r"Redshift $z$", fontsize=8)
        if col == 0:
            ax.set_ylabel(r"$D_{\rm KL}$ [nats]", fontsize=8)
            ax.annotate(
                row_titles[row],
                xy=(-0.5, 0.5),
                xycoords="axes fraction",
                fontsize=8,
                rotation=90,
                va="center",
                ha="center",
            )
        if col == 3:
            ax.legend(
                fontsize=6,
                loc="upper right",
            )

plt.tight_layout()
plt.savefig(
    "lbi/figs/redshift_dependence_dkl.png", dpi=300, bbox_inches="tight"
)
plt.close()
