"""Plot heatmap of sigma_MS 10/100 Myr ratio across redshift and mass bins."""

import matplotlib.pyplot as plt
import numpy as np
import yaml

with open("observations/intrinsic_scatter_per_timescale_errors.yaml") as f:
    obs = yaml.safe_load(f)

plt.rcParams.update({"font.size": 8})

masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
red_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]
mass_labels = ["8-8.5", "8.5-9", "9-9.5", "9.5-10"]
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}

n_mass = len(masses)
n_z = len(red)
ratio = np.full((n_mass, n_z), np.nan)

for i, mass in enumerate(masses):
    for j, z in enumerate(red):
        target = f"{mass}_{z}"
        if target in skips or target not in obs:
            continue
        d = obs[target]["data"]
        ratio[i, j] = d[0] / d[5]

# Also print table
print(f"{'Target':15s}  {'10/100 Myr':>10}  data (10,20,30,40,50,100 Myr)")
print("-" * 75)
for j, z in enumerate(red):
    for i, mass in enumerate(masses):
        target = f"{mass}_{z}"
        if np.isnan(ratio[i, j]) or target not in obs:
            continue
        print(f"{target:15s}  {ratio[i,j]:>10.2f}  {obs[target]['data']}")

fig, ax = plt.subplots(figsize=(4, 3))
vmin, vmax = np.nanmin(ratio), np.nanmax(ratio)
im = ax.pcolormesh(
    np.arange(n_mass), np.arange(n_z), ratio.T,
    cmap="plasma", vmin=vmin, vmax=vmax,
)
for i in range(n_mass):
    for j in range(n_z):
        if not np.isnan(ratio[i, j]):
            ax.text(
                i, j, f"{ratio[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if ratio[i, j] < 0.6 * vmax else "black",
            )

ax.set_xticks(np.arange(n_mass))
ax.set_xticklabels(mass_labels, rotation=45, ha="right")
ax.set_yticks(np.arange(n_z))
ax.set_yticklabels(red_labels)
ax.set_ylabel("Redshift")
ax.set_xlabel(r"$\log M_* / M_\odot$")
ax.invert_yaxis()
fig.colorbar(im, ax=ax, label=r"$\sigma_{MS}(10\,\mathrm{Myr})\,/\,\sigma_{MS}(100\,\mathrm{Myr})$")
fig.suptitle(r"Ratio of $\sigma_{MS}$ at 10 vs 100 Myr", fontsize=9)
plt.tight_layout()
plt.savefig("lbi/sigma_ms_shape.png", dpi=300, bbox_inches="tight")
plt.close()
