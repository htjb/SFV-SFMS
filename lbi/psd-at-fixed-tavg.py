"""Plot the PSDs inferred from LBI fits at a fixed timescale."""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM

from burstiness.utils import calc_extreg_psd, calc_sho_psd

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

plt.rcParams.update({"font.size": 8})

redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
latex_m = [
    r"$8 \leq \log M_* / M_\odot \leq 8.5$",
    r"$8.5 < \log M_* / M_\odot \leq 9$",
    r"$9 < \log M_* / M_\odot \leq 9.5$",
    r"$9.5 < \log M_* / M_\odot \leq 10$",
]
latex_z = [r"$3 \leq z \leq 4$", r"$4 < z \leq 5$", r"$5 < z \leq 6$", r"$6 < z \leq 7$", r"$7 < z \leq 8$"]
mass_bins = {
    "M_A": "8.0_8.5",
    "M_B": "8.5_9.0",
    "M_C": "9.0_9.5",
    "M_D": "9.5_10.0",
}
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}

with open(
    "observations/intrinsic_scatter_per_timescale_errors.yaml",
) as f:
    data = yaml.safe_load(f)

kernels = ["sho", "ext-reg"]
nsamples = 10000
nparams = 2500

results = {"sho": {}, "ext-reg": {}}
for kernel in kernels:
    ndims = 5 if kernel == "ext-reg" else 3
    nlive = ndims * 250

    calc_psd = calc_sho_psd if kernel == "sho" else calc_extreg_psd

    if kernel == "sho":
        pnames = ["s0", "tau0", "Q"]
    elif kernel == "ext-reg":
        pnames = ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]

    psd_30Myrs = {}
    psd_10Myrs = {}
    psd_50Myrs = {}
    evidences = {}
    for i, mass in enumerate(masses):
        psd_30Myrs_z = {}
        psd_10Myrs_z = {}
        psd_50Myrs_z = {}
        evidences_z = {}
        for j, (z_val, z) in enumerate(zip(redshifts, red)):
            target = f"{mass}_{z}"
            if target in skips:
                psd_30Myrs_z[z_val] = [np.nan, np.nan, np.nan]
                psd_10Myrs_z[z_val] = [np.nan, np.nan, np.nan]
                psd_50Myrs_z[z_val] = [np.nan, np.nan, np.nan]
                evidences_z[z_val] = np.nan
                continue

            mb = mass_bins[mass]
            chains_dir = (
                f"lbi/chains/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
                f"-{nparams}-{nsamples}_z_{z_val}_mass_bin_{mb}_nlive_{nlive}"
            )
            path = f"{chains_dir}/{target}/posterior.npz"
            theta = 10 ** np.load(path)["samples_log10"]  # physical space
            evidence = np.loadtxt(f"{chains_dir}/{target}/logZ.txt")
            evidences_z[z_val] = evidence

            mean_psd = np.array([
                calc_psd(np.array([0.01, 0.03, 0.05]), th) for th in theta
            ])

            psd_10Myrs_z[z_val] = [
                np.median(mean_psd[:, 0]),
                np.percentile(mean_psd[:, 0], 16),
                np.percentile(mean_psd[:, 0], 84),
            ]
            psd_30Myrs_z[z_val] = [
                np.median(mean_psd[:, 1]),
                np.percentile(mean_psd[:, 1], 16),
                np.percentile(mean_psd[:, 1], 84),
            ]
            psd_50Myrs_z[z_val] = [
                np.median(mean_psd[:, 2]),
                np.percentile(mean_psd[:, 2], 16),
                np.percentile(mean_psd[:, 2], 84),
            ]
        psd_30Myrs[mass] = psd_30Myrs_z
        psd_10Myrs[mass] = psd_10Myrs_z
        psd_50Myrs[mass] = psd_50Myrs_z
        evidences[mass] = evidences_z

    results[kernel]["30Myrs"] = psd_30Myrs
    results[kernel]["10Myrs"] = psd_10Myrs
    results[kernel]["50Myrs"] = psd_50Myrs
    results[kernel]["evidences"] = evidences

cmap = plt.get_cmap("inferno_r")
timescales = ["10"]#, "30", "50"]
offset = [-0.1, 0.1]  # horizontal offset for visibility
markers = ["o", "s"]  # different markers for kernels

for t in timescales:
    fig, ax = plt.subplots(1, 4, figsize=(10, 5), sharey=True, sharex=True)

    # Compute global logK range across all kernels/masses for this timescale
    all_logK = []
    for kernel in kernels:
        alt_kernel = "ext-reg" if kernel == "sho" else "sho"
        for mass in masses:
            for z_val in redshifts:
                all_logK.append(
                    results[kernel]["evidences"][mass][z_val]
                    - results[alt_kernel]["evidences"][mass][z_val]
                )
    vmin, vmax = np.nanmin(all_logK), np.nanmax(all_logK)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, kernel in enumerate(kernels):
        psd_Myrs = results[kernel][t + "Myrs"]
        alt_kernel = "ext-reg" if kernel == "sho" else "sho"
        for j, mass in enumerate(masses):
            z_vals, psd_medians, psd_errs, logK = [], [], [], []
            for z_val in redshifts:
                z_vals.append(z_val)
                median, err_low, err_high = psd_Myrs[mass][z_val]
                psd_medians.append(median)
                psd_errs.append([[median - err_low], [err_high - median]])
                logK.append(
                    results[kernel]["evidences"][mass][z_val]
                    - results[alt_kernel]["evidences"][mass][z_val]
                )
            for z, med, err, lk in zip(z_vals, psd_medians, psd_errs, logK):
                markerfacecolor = cmap(norm(lk)) if kernel == "sho" else 'none'
                ax[j].errorbar(
                    [z + offset[i]], [med],
                    yerr=np.array(err),
                    color=cmap(norm(lk)),   # norm maps raw logK -> [0,1]
                    capsize=5,
                    linestyle="-",
                    marker=markers[i],
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=cmap(norm(lk))
                )
            ax[j].set_xlabel("Redshift")
            ax[j].set_title(f"{latex_m[j]}")
            ax[j].set_yscale("log")
            ax[j].set_xticks(redshifts)
            ax[j].grid(True, which='major', ls="--", alpha=0.5)
    ax[0].set_ylabel(f"PSD at {t} Myr")

    # ScalarMappable ties the cmap + norm together so fig.colorbar knows the scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required but unused; the norm carries the data range
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\log K$", fontsize=8)

    # after the kernel/mass loops, before tight_layout: 
    legend_handles = [
        mlines.Line2D([], [], color='grey', marker='o',
              markerfacecolor='grey', linestyle='-', label=kernels[0]),
        mlines.Line2D([], [], color='grey', marker='s',
                markerfacecolor='none', markeredgecolor='grey',
                linestyle='-', label=kernels[1])
    ]
    ax[0].legend(handles=legend_handles)
    plt.savefig(
        f"lbi/figs/psd_fixed_tavg_{t}Myrs_function_of_z.png",
        dpi=300,
        bbox_inches="tight",
    )
    #plt.show()
    plt.close()

# now with mass on the x-axis
for t in timescales:
    fig, ax = plt.subplots(1, 5, figsize=(8, 4), sharey=True, sharex=True)

    all_logK = []
    for kernel in kernels:
        alt_kernel = "ext-reg" if kernel == "sho" else "sho"
        for mass in masses:
            for z_val in redshifts:
                all_logK.append(
                    results[kernel]["evidences"][mass][z_val]
                    - results[alt_kernel]["evidences"][mass][z_val]
                )
    vmin, vmax = np.nanmin(all_logK), np.nanmax(all_logK)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, kernel in enumerate(kernels):
        psd_Myrs = results[kernel][t + "Myrs"]
        alt_kernel = "ext-reg" if kernel == "sho" else "sho"
        for j, z_val in enumerate(redshifts):
            for mass, mass_val in zip(masses, [8.25, 8.75, 9.25, 9.75]):
                median, err_low, err_high = psd_Myrs[mass][z_val]
                lk = (results[kernel]["evidences"][mass][z_val]
                      - results[alt_kernel]["evidences"][mass][z_val])
                markerfacecolor = cmap(norm(lk)) if kernel == "sho" else 'none'
                ax[j].errorbar(
                    [mass_val + offset[i] * 0.5], [median],
                    yerr=np.array([[[median - err_low], [err_high - median]]]).T[0, :, :],
                    color=cmap(norm(lk)),
                    capsize=5, marker=markers[i], linestyle="-",
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=cmap(norm(lk)),
                )
            ax[j].set_xticks([8.25, 8.75, 9.25, 9.75])
            ax[j].set_xticklabels(latex_m, rotation=90)
            ax[j].set_xlabel(r"$\log M_* / M_\odot$")
            ax[j].set_title(latex_z[j])
            ax[j].set_yscale("log")
            ax[j].grid(True, which='major', ls="--", alpha=0.5)
    ax[0].set_ylabel(rf"PSD ($\tau = {t}$ [Myr]) [dex$^2$ Gyr]")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\log K$", fontsize=8)

    legend_handles = [
        mlines.Line2D([], [], color='grey', marker='o',
                      markerfacecolor='grey', linestyle='-', label="SHO"),
        mlines.Line2D([], [], color='grey', marker='s',
                      markerfacecolor='none', markeredgecolor='grey',
                      linestyle='-', label="ExtReg"),
    ]
    ax[0].legend(handles=legend_handles)
    plt.savefig(
        f"lbi/figs/psd_fixed_tavg_{t}Myrs_function_of_mass.png",
        dpi=300, bbox_inches="tight",
    )
    plt.show()
