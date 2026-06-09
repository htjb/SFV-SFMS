"""Plot histogram summary of emulator LBI results."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from anesthetic.plot import kde_plot_1d

from burstiness.prior import prior as prior_dict

plt.rcParams.update({"font.size": 10})

kernel = sys.argv[1]  # "ext-reg" or "sho"
nsamples = 10000
ndims = 5 if kernel == "ext-reg" else 3
nlive = ndims * 250

masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
z_vals = [3.5, 4.5, 5.5, 6.5, 7.5]
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}
mass_bins = {
    "M_A": "8.0_8.5", "M_B": "8.5_9.0",
    "M_C": "9.0_9.5", "M_D": "9.5_10.0",
}

if kernel == "ext-reg":
    param_names = ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]
    latex_names = [
        r"$\log \sigma_\mathrm{Reg}$",
        r"$\log \tau_\mathrm{eq}$",
        r"$\log \tau_\mathrm{in}$",
        r"$\log \sigma_\mathrm{Dyn}$",
        r"$\log \tau_\mathrm{dyn}$",
    ]
else:
    param_names = ["s0", "tau0", "Q"]
    latex_names = [r"$\log S_{0}$", r"$\log \tau_0$", r"$\log Q$"]

# Prior samples in log10 space
log10_prior_lo = np.array(
    [np.log10(prior_dict[p]["min"]) for p in param_names]
)
log10_prior_hi = np.array(
    [np.log10(prior_dict[p]["max"]) for p in param_names]
)
rng = np.random.default_rng(0)
prior_samples = (
    rng.uniform(size=(2000, ndims))
    * (log10_prior_hi - log10_prior_lo)
    + log10_prior_lo
)

# Load posterior samples from posterior.npz
params = {}
for mass in masses:
    for z, red_tag in zip(z_vals, red):
        target = f"{mass}_{red_tag}"
        if target in skips:
            params[target] = None
            continue
        folder = (
            f"lbi/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
            f"-2500-{nsamples}_z_{z}_mass_bin_{mass_bins[mass]}"
            f"_nlive_{nlive}"
        )
        path = f"{folder}/{target}/posterior.npz"
        try:
            npz = np.load(path)
            samples = npz["samples_log10"]   # (n_samples, ndims)
            params[target] = {
                p + "_samples": samples[:, i]
                for i, p in enumerate(param_names)
            }
        except FileNotFoundError:
            print(f"File not found: {path}")
            params[target] = None


redshifts = [
    r"$3 \leq z \leq 4$",
    r"$4 < z \leq 5$",
    r"$5 < z \leq 6$",
    r"$6 < z \leq 7$",
    r"$7 < z \leq 8$",
]
latex_m = [
    r"A: $8 \leq \log M_* / M_\odot \leq 8.5$",
    r"B: $8.5 < \log M_* / M_\odot \leq 9$",
    r"C: $9 < \log M_* / M_\odot \leq 9.5$",
    r"D: $9.5 < \log M_* / M_\odot \leq 10$",
]
colors = ["blue", "orange", "green", "red", "purple"]

fig, axes = plt.subplots(
    ndims, len(masses), figsize=(6.3, len(latex_names) * 1.5), sharey="row"
)
for i, pname in enumerate(param_names):
    for j, mass in enumerate(masses):
        # Prior
        kde_plot_1d(
            axes[i, j], prior_samples[:, i], color="gray", label="Prior"
        )
        for k, (red_tag, zlabel) in enumerate(zip(red, redshifts)):
            target = f"{mass}_{red_tag}"
            if params[target] is None:
                continue
            kde_plot_1d(
                axes[i, j],
                params[target][pname + "_samples"],
                color=colors[k],
                label=zlabel,
            )
        axes[i, j].set_xlabel(latex_names[i])

for j in range(len(masses)):
    axes[0, j].set_title(latex_m[j], fontsize=8)

for i in range(ndims):
    axes[i, 0].set_yticks([])

handles = [
    plt.Line2D([0], [0], color=colors[k], label=redshifts[k])
    for k in range(len(redshifts))
]
handles.append(plt.Line2D([0], [0], color="grey", label="Prior"))

plt.tight_layout()
plt.subplots_adjust(wspace=0)
bbox_y = 1.07 if ndims == 5 else 1.15
fig.legend(
    handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, bbox_y)
)
plt.savefig(
    f"lbi/{kernel}_hist_plot.png", dpi=300, bbox_inches="tight"
)
plt.close()
