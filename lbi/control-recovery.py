"""Plot recovered vs true kernel parameters for every control run.

The control mock observations are built in blackjax-inference.py from row
``control_indices[control]`` of each emulator's held-out test set, so the
true parameters are recovered from ``test_params.txt`` in the matching
emulator directory.

Usage:
    python lbi/control-recovery.py [chains_root] [emulators_root] [out_dir]
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from burstiness.prior import prior as prior_dict

plt.rcParams.update({"font.size": 8})

chains_root = sys.argv[1] if len(sys.argv) > 1 else "chains-for-zenodo"
emulators_root = sys.argv[2] if len(sys.argv) > 2 else "emulators-for-zenodo"
out_dir = sys.argv[3] if len(sys.argv) > 3 else "."

# Must match blackjax-inference.py
control_indices = {"control": 0, "control2": 61, "control3": 4}

pnames = {
    "sho": ["s0", "tau0", "Q"],
    "ext-reg": ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"],
}
plabels = {
    "s0": r"$\log S_0$",
    "tau0": r"$\log \tau_0$",
    "Q": r"$\log Q$",
    "sig_reg": r"$\log \sigma_\mathrm{reg}$",
    "tau_eq": r"$\log \tau_\mathrm{eq}$",
    "tau_in": r"$\log \tau_\mathrm{in}$",
    "sig_dyn": r"$\log \sigma_\mathrm{dyn}$",
    "tau_dyn": r"$\log \tau_\mathrm{dyn}$",
}

redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
z_colors = dict(zip(redshifts, plt.get_cmap("viridis")(np.linspace(0, 0.9, 5))))
mass_markers = {"8.0_8.5": "o", "8.5_9.0": "s", "9.0_9.5": "^", "9.5_10.0": "D"}

dir_pattern = re.compile(
    r"blackjax_chains_astroemu-(?P<kernel>sho|ext-reg)-(?P<rest>.+)"
    r"_z_(?P<z>[\d.]+)_mass_bin_(?P<mb>[\d.]+_[\d.]+)_nlive_\d+$"
)


def collect(kernel):
    """Gather truth and posterior summaries for every control of a kernel.

    Returns:
        List of dicts with keys z, mb, control, truth, median, lo, hi
        (all parameter arrays in log10 space, shape (ndims,)) and rank,
        the fraction of posterior samples below the truth per parameter.
    """
    rows = []
    for chains_dir in sorted(glob.glob(f"{chains_root}/blackjax_chains_*")):
        m = dir_pattern.match(os.path.basename(chains_dir))
        if m is None or m["kernel"] != kernel:
            continue
        emulator_dir = os.path.join(
            emulators_root,
            f"burstiness-emulator-astroemu-{kernel}-{m['rest']}"
            f"_z_{m['z']}_mass_bin_{m['mb']}",
        )
        test_params_phys = np.loadtxt(os.path.join(emulator_dir, "test_params.txt"))

        for control, idx in control_indices.items():
            paths = glob.glob(f"{chains_dir}/*/{control}/posterior.npz")
            if not paths:
                print(f"missing {control} in {chains_dir}")
                continue
            # Samples are equally weighted (importance-resampled in inference)
            samples = np.load(paths[0])["samples_log10"]
            lo, med, hi = np.percentile(samples, [16, 50, 84], axis=0)
            truth = np.log10(test_params_phys[idx])
            rows.append({
                "z": float(m["z"]),
                "mb": m["mb"],
                "control": control,
                "truth": truth,
                "median": med,
                "lo": lo,
                "hi": hi,
                "rank": np.mean(samples < truth, axis=0),
            })
    return rows


def print_coverage(kernel, names, rows):
    """Print the fraction of controls whose truth lies in the central 68%/95%
    credible intervals, with the binomial standard error for comparison."""
    ranks = np.array([r["rank"] for r in rows])  # (n_controls, ndims)
    n = len(ranks)
    print(f"\n{kernel}: coverage over {n} control runs")
    for level in (0.68, 0.95):
        err = np.sqrt(level * (1 - level) / n)
        print(f"  nominal {level:.2f} (binomial s.e. {err:.3f})")
        tail = (1 - level) / 2
        inside = (ranks > tail) & (ranks < 1 - tail)
        for i, p in enumerate(names):
            print(f"    {p:8s} {inside[:, i].mean():.2f}")


for kernel, names in pnames.items():
    rows = collect(kernel)
    ndims = len(names)
    ncols = ndims + 1 if ndims <= 3 else 3
    nrows = int(np.ceil((ndims + 1) / ncols))  # +1 leaves room for the legend
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.3, 2.2 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for i, p in enumerate(names):
        ax = axes[i]
        bounds = np.log10([prior_dict[p]["min"], prior_dict[p]["max"]])
        ax.plot(bounds, bounds, c="gray", lw=1, ls="--", zorder=0)
        for r in rows:
            ax.errorbar(
                r["truth"][i], r["median"][i],
                yerr=[[r["median"][i] - r["lo"][i]], [r["hi"][i] - r["median"][i]]],
                fmt=mass_markers[r["mb"]], color=z_colors[r["z"]],
                ms=4, mec="white", mew=0.5, elinewidth=0.8, capsize=0,
                alpha=0.85,
            )
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal")
        ax.set_xlabel(f"True {plabels[p]}")
        ax.set_ylabel(f"Recovered {plabels[p]}")

    for ax in axes[ndims:]:
        ax.axis("off")

    handles = [
        Line2D([], [], ls="", marker="o", color=z_colors[z], label=f"$z={z}$")
        for z in redshifts
    ] + [
        Line2D([], [], ls="", marker=mk, color="gray",
               label=rf"$\log M_\star \in [{mb.replace('_', ', ')}]$")
        for mb, mk in mass_markers.items()
    ]
    axes[-1].legend(handles=handles, loc="center", fontsize=7, frameon=False)

    print_coverage(kernel, names, rows)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"control_recovery_{kernel}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
