"""Calculate the DKL for different emulator errors."""

import numpy as np
import matplotlib.pyplot as plt
import yaml

with open(
    "observations/intrinsic_scatter_per_timescale_errors.yaml",
) as f:
    rawdata = yaml.safe_load(f)

targets = rawdata.keys()
targets = [
    target for target in targets if not ("M_E" in target or "M_F" in target)
]

def kl_divergence(eerror: float, relevant_bin: str) -> float:
    """Calculate the DKL for a given emulator error."""
    data_error = np.mean(
        [rawdata[relevant_bin]["high_error"], rawdata[relevant_bin]["low_error"]]
    )
    if data_error < 0.03:
        data_error = 0.03

    emulator_error = rawdata[relevant_bin]["data"] * eerror / 100

    eerror = np.mean(emulator_error)
    kl = 6 / 2 * (eerror / np.mean(data_error)) ** 2
    print(f"Bin: {relevant_bin}, DKL: {kl}, Error: {eerror}")
    return kl


kernels = ["ext-reg", "sho"]
mass_bins = ["8.0_8.5", "8.5_9", "9_9.5", "9.5_10"]
nsamples = 10000
redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
mapping = {
    "8.0_8.5": "M_A", "8.5_9.0": "M_B",
    "9.0_9.5": "M_C", "9.5_10.0": "M_D",
}
redshift_mapping = {
    3.5: "z3_4", 4.5: "z4_5", 5.5: "z5_6",
    6.5: "z6_7", 7.5: "z7_8",
}

skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}

all_errors = {}
for kernel in kernels:
    mean_errors, ninefive_errors = [], []
    for mass_bin in mass_bins:
        for z in redshifts:
            mass_bin_tag = (
                str(float(mass_bin.split("_")[0]))
                + "_"
                + str(float(mass_bin.split("_")[1]))
            )
            relevant_bin = (
                f"{mapping[mass_bin_tag]}_{redshift_mapping[z]}"
            )
            if relevant_bin in skips:
                mean_errors.append(np.nan)
                ninefive_errors.append(np.nan)
                continue
            file_tag = (
                f"{kernel}-sfr_theory_mean-2500-{nsamples}"
                f"_z_{z}_mass_bin_{mass_bin_tag}"
            )
            emulator_dir = (
                "lbi/fit_burstiness_project/"
                "burstiness-emulator-astroemu-" + file_tag + "/"
            )
            mean_emulator_error = np.loadtxt(
                emulator_dir + "mean_error.txt"
            )
            ninefive_emulator_error = np.loadtxt(
                emulator_dir + "95th_percentile_error.txt"
            )
            mean_errors.append(
                kl_divergence(mean_emulator_error, relevant_bin)
            )
            ninefive_errors.append(
                kl_divergence(ninefive_emulator_error, relevant_bin)
            )
    all_errors[kernel] = (mean_errors, ninefive_errors)

mass_lo = [float(mb.split("_")[0]) for mb in mass_bins]

mass_labels = ["A:[8, 8.5]", "B:(8.5, 9]", "C:(9, 9.5]", "D:(9.5, 10]"]
red_labels = ["[3, 4]", "(4, 5]", "(5, 6]", "(6, 7]", "(7, 8]"]

all_vals = [v for me, _ in all_errors.values() for v in me if not np.isnan(v)]
vmin, vmax = min(all_vals), max(all_vals)

kernel_labels = {"ext-reg": r"ExtReg", "sho": r"SHO"}
fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True, sharey=True)

for row, kernel in enumerate(kernels):
    mean_errors, _ = all_errors[kernel]
    ax = axes[row]
    grid = np.array(mean_errors).reshape(len(mass_bins), len(redshifts))
    im = ax.pcolormesh(
        np.array(redshifts),
        np.array(mass_lo),
        grid,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    for i, z in enumerate(redshifts):
        for j, m in enumerate(mass_lo):
            if np.isnan(grid[j, i]):
                continue
            color = "white" if grid[j, i] < (vmin + vmax) / 2 else "black"
            ax.text(
                z, m, f"{grid[j, i]:.2f}",
                ha="center", va="center", fontsize=8, color=color,
            )
    ax.set_title(kernel_labels[kernel])
    ax.set_ylabel(r"$\log_{10} M_*/M_\odot$")
    ax.set_xticks(redshifts, labels=red_labels)
    ax.set_yticks(mass_lo)
    ax.set_yticklabels(mass_labels)
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right")

axes[0].invert_yaxis()
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
axes[1].set_xlabel(r"Redshift $z$")

fig.subplots_adjust(
    left=0.25, right=0.78, top=0.95, bottom=0.15, hspace=0.2,
)
cbar_ax = fig.add_axes((0.81, 0.08, 0.03, 0.87))
fig.colorbar(im, cax=cbar_ax, label=r"$D_{KL} (P || P_\epsilon)$")
plt.savefig("lbi/dkl_emulator_error.png", dpi=300)
plt.close(fig)

