"""Compare evidence between extended regulator and SHO models."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8})

z_vals = [3.5, 4.5, 5.5, 6.5, 7.5]
z_tags = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
m_tags = ["M_A", "M_B", "M_C", "M_D"]
mass_bins = {"M_A": "8.0_8.5", "M_B": "8.5_9.0", "M_C": "9.0_9.5", "M_D": "9.5_10.0"}
nsamples = 10000
nlive = {"ext-reg": 1250, "sho": 750}

bound_m = ["A:[8, 8.5]", "B:(8.5, 9]", "C:(9, 9.5]", "D:(9.5, 10]"]
bound_z = ["[3, 4]", "(4, 5]", "(5, 6]", "(6, 7]", "(7, 8]"]
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}


def load_logz(kernel, z, mass_tag, red_tag):
    mb = mass_bins[mass_tag]
    folder = (
        f"lbi/chains/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
        f"-2500-{nsamples}_z_{z}_mass_bin_{mb}_nlive_{nlive[kernel]}"
    )
    path = f"{folder}/{mass_tag}_{red_tag}/logZ.txt"
    return float(np.loadtxt(path))


dlz, evidenceA, evidenceB = [], [], []
for z, red in zip(z_vals, z_tags):
    dlz_m, evidenceA_m, evidenceB_m = [], [], []
    for mass in m_tags:
        if f"{mass}_{red}" in skips:
            dlz_m.append(np.nan)
            evidenceA_m.append(np.nan)
            evidenceB_m.append(np.nan)
            continue
        try:
            logZ_A = load_logz("ext-reg", z, mass, red)
            logZ_B = load_logz("sho", z, mass, red)
            dlz_m.append(logZ_A - logZ_B)
            evidenceA_m.append(logZ_A)
            evidenceB_m.append(logZ_B)
        except FileNotFoundError:
            print(f"File not found for {mass} and {red}.")
            dlz_m.append(np.nan)
            evidenceA_m.append(np.nan)
            evidenceB_m.append(np.nan)

    dlz.append(dlz_m)
    evidenceA.append(evidenceA_m)
    evidenceB.append(evidenceB_m)

dlz = np.array(dlz)
evidenceA = np.array(evidenceA)
evidenceB = np.array(evidenceB)

mass_labels = [
    r"$8.0 \leq \log M_* / M_\odot < 8.5$",
    r"$8.5 \leq \log M_* / M_\odot < 9.0$",
    r"$9.0 \leq \log M_* / M_\odot < 9.5$",
    r"$9.5 \leq \log M_* / M_\odot < 10.0$",
]

latex_z = [r"$3 < z \leq 4$", r"$4 < z \leq 5$", r"$5 < z \leq 6$", r"$6 < z \leq 7$", r"$7 < z \leq 8$"]

fig, axes = plt.subplots(1, 4, figsize=(7, 2.5), sharey=True, sharex=True)

x = np.arange(len(z_vals))

for i, (ax, mlabel) in enumerate(zip(axes, mass_labels)):
    heights = dlz[:, i]
    colors = [
        "#4C72B0" if (np.isnan(h) or h >= 0) else "#DD8452"
        for h in heights
    ]
    bars = ax.bar(x, heights, color=colors, linewidth=0.5, width=0.7, zorder=3)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="-")
    # for thresh, ls in [(1, "--"), (3, ":")]:
    #     ax.axhline(thresh, color="grey", linewidth=0.6, linestyle=ls)
    #     ax.axhline(-thresh, color="grey", linewidth=0.6, linestyle=ls)
   
    ax.set_xticks(x)
    ax.set_xticklabels(latex_z, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(r"Redshift $z$", fontsize=8)
    ax.set_title(mlabel, fontsize=7)
    if i == 0:
        ax.set_ylabel(
            r"$\log Z_{\rm ExtReg} - \log Z_{\rm SHO}$", fontsize=8
        )

    ax.grid(True, linestyle="--", alpha=0.5, zorder=-1)

plt.tight_layout()
plt.savefig("lbi/figs/evidence_ratio.png", dpi=300)
plt.close()


