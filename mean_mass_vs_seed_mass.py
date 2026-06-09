"""code to test the mean function during development."""

import astropy.units as u
#import jax.numpy as jnp
import numpy as jnp
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, z_at_value

from burstiness.gp_sfh_jax import get_tarr
from burstiness.mean_function import (
    sfr_theoretical_mean,
)
import tqdm

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

zvals = jnp.linspace(3, 8, 20)
mhalo_seeds = jnp.logspace(6.5, 9.5, 25)
# mhalo_seeds = [
#     10**(6.5),
#     1e7,
#     10**(7.5),
#     1e8,
#     10**(8.5),
#     1e9,
#     10**(9.5)
# ]

stellar_masses = []
plot_stellar_masses = []
for i in tqdm.tqdm(range(len(zvals))):
    zval = zvals[i]
    smm = []
    psmm = []
    for mhalo_seed in mhalo_seeds:
        tarr = get_tarr(n_tarr=50, zval=float(zval))
        z = [z_at_value(cosmo.age, t * u.Gyr).value for t in tarr]
        sfr, halo_mass, stellar_mass = sfr_theoretical_mean(
            tarr, mhalo_seed=mhalo_seed
        )
        stellar_mass = jnp.log10(stellar_mass)

        if stellar_mass < 7 or stellar_mass > 11:
            psmm.append(jnp.nan)  # mark out-of-range values as NaN for plotting
        else:
            psmm.append(stellar_mass)
        smm.append(stellar_mass)
    stellar_masses.append(smm)
    plot_stellar_masses.append(psmm)
stellar_masses = jnp.array(stellar_masses)
plot_stellar_masses = jnp.array(plot_stellar_masses)
mhalo_seeds = jnp.array(mhalo_seeds)

import matplotlib.cm as cm
import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(3.5, 2.5))

cmap = cm.viridis_r
norm = mcolors.Normalize(vmin=min(zvals), vmax=max(zvals))

for i, zval in enumerate(zvals):
    ax.plot(
        jnp.log10(mhalo_seeds),
        plot_stellar_masses[i],
        color=cmap(norm(zval)),
    )

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label=r"Redshift $z$")
cbar.ax.invert_yaxis()

ax.set_xlabel(r"$\log_{10}(M_h^{\rm seed} / M_\odot)$")
ax.set_ylabel(r"$\log_{10}(M_* / M_\odot)$")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(
    "final_stellar_mass_vs_seed_halo_mass_and_redshift.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ---------------------------------------------------------------------------
# Interpolate to find seed masses for target (z, stellar mass bin centre)
# ---------------------------------------------------------------------------
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
import numpy as np

log10_seeds = np.log10(np.array(mhalo_seeds))
interp = RegularGridInterpolator(
    (np.array(zvals), log10_seeds),
    np.array(stellar_masses),
    method="linear",
)

target_zvals = [3.5, 4.5, 5.5, 6.5, 7.5]
target_bin_centers = [8.25, 8.75, 9.25, 9.75]

print("\n# Seed masses for data-gen.sh SEED_MASS entries:")
print("# z     M*_bin_center   log10(seed)   seed_mass")
for z in target_zvals:
    for m_star in target_bin_centers:
        f = lambda log10_seed: interp([[z, log10_seed]])[0] - m_star
        lo, hi = log10_seeds[0], log10_seeds[-1]
        if f(lo) * f(hi) < 0:
            root = brentq(f, lo, hi)
            print(f"  z={z}  M*={m_star}  ->  log10(seed)={root:.2f}  seed={10**root:.3e}")
        else:
            print(f"  z={z}  M*={m_star}  ->  OUT OF GRID RANGE (extend mhalo_seeds)")
