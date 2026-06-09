import jax
import jax.numpy as jnp
import matplotlib.patches as rect
import matplotlib.pyplot as plt
import yaml
from astropy.io import fits
from astropy.table import Table


redshifts = [[3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
masses = [[8, 8.5], [8.5, 9], [9, 9.5], [9.5, 10]]

bound_m = ["A: [8, 8.5]", "B: (8.5, 9]", "C: (9, 9.5]", "D: (9.5, 10]"]
bound_z = ["[3, 4]", "(4, 5]", "(5, 6]", "(6, 7]", "(7, 8]"]

latex_z = [
    r"$3 \leq z \leq 4$",
    r"$4 < z \leq 5$",
    r"$5 < z \leq 6$",
    r"$6 < z \leq 7$",
    r"$7 < z \leq 8$",
]
latex_m = [ r"$8 \leq \log M_* / M_\odot \leq 8.5$",
    r"$8.5 < \log M_* / M_\odot \leq 9$",
    r"$9 < \log M_* / M_\odot \leq 9.5$",
    r"$9.5 < \log M_* / M_\odot \leq 10$",
]


# Observation data
# ---------------------------------------------------------------------------
with open("observations/intrinsic_scatter_per_timescale_errors.yaml") as f:
    obs_data = yaml.safe_load(f)


mass_keys = ['M_A', 'M_B', 'M_C', 'M_D']
redshift_keys = ['z3_4', 'z4_5', 'z5_6', 'z6_7', 'z7_8']
keys = [f"{mass}_{redshift}" for mass in mass_keys for redshift in redshift_keys]

skips = ['M_C_z7_8', 'M_D_z7_8', 'M_D_z6_7']  # Skip bins with insufficient data

# ----------------------------------------------------------------------------
# Plot the intrinsic scatter of the SFR-M* relation as a function of timescale 
# for each bin of mass and redshift
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(10, 6))
ax = ax.flatten()

timescales = jnp.array([10, 20, 30, 40, 50, 100]) * 1e-3  # in Myr

for i, rkey in enumerate(redshift_keys):
    for j, mkey in enumerate(mass_keys):
        key = f"{mkey}_{rkey}"
        print(key, latex_z[i])
        if key in obs_data and key not in skips:
            scatter = obs_data[key]['data']
            low_error = obs_data[key]['low_error']
            high_error = obs_data[key]['high_error']
            ax[i].errorbar(timescales, scatter, yerr=(low_error, high_error), fmt='o', capsize=5,
                           linestyle='-', label=f"{bound_m[j]}") 
    
    ax[i].set_xscale('log')
    if i > 2:
        ax[i].set_xlabel('Timescale [Gyr]')
    #ax[i].set_xticks(ax[3].get_xticks(), labels=['']*len(ax[3].get_xticks()))
    if i in [0, 3]:
        ax[i].set_ylabel(r'$\sigma_{\mathrm{MS}}$ [dex]')
    #else:
    ax[i].set_title(f"{latex_z[i]}")

ylims = [ax[i].get_ylim() for i in range(5)]
ymin = min(ylim[0] for ylim in ylims)
ymax = max(ylim[1] for ylim in ylims)

for i in range(5):
    ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[i].set_xticks(jnp.array([10, 30, 50, 100])*1e-3, labels=[r"$0.01$", r"$0.03$", r"$0.05$", r"$0.1$"])

    ax[i].set_ylim((ymin, ymax))
    ax[i].set_yticks(ax[0].get_yticks())

ax[0].legend(loc='upper right', fontsize='small')

# ----------------------------------------------------------------------------
# Plot the number of galaxies in each bin of mass and redshift
# ---------------------------------------------------------------------------

# Open the FITS file
hdulist = fits.open("observations/full_sample_combined_not_logged_ELS.fits")

# Access the primary data (image, table, etc.)
data = hdulist[1].data
header = hdulist[1].header
data = Table(data)
df = data.to_pandas()
mass = df['log(M)']
redshift = df['z']


# Example: store results in a dictionary
binned_galaxies = {}

numbers = []
for i, (zmin, zmax) in enumerate(redshifts):
    numbers_m = []
    for j, (mmin, mmax) in enumerate(masses):
        mask = (redshift >= zmin) & (redshift < zmax) & (mass >= mmin) & (mass < mmax)
        key = f"{mass_keys[j]}_{redshift_keys[i]}"
        binned_galaxies[key] = df.loc[mask,['log(M)', 'z', 'SFR100', 'SFR10']].to_dict(orient='list')
        numbers_m.append(len(binned_galaxies[key]['z']))
    numbers.append(numbers_m)

# Convert numbers to a 2D array for heatmap
numbers_array = jnp.array(numbers)
total = jnp.sum(numbers_array)

cp = ax[-1].imshow(numbers_array, cmap='viridis', aspect='auto')
# Add text overlay
for i in range(len(redshifts)):
    for j in range(len(masses)):
        ax[-1].text(j, i, numbers_array[i, j],
                 ha="center", va="center", 
                 color="white" if numbers_array[i, j] < 2000 else "black")
        # Add red border if count < 30
        if numbers_array[i, j] < 30:
            ax[-1].add_patch(rect.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                               fill=False, edgecolor='red', lw=2))
plt.colorbar(cp, label='Number of Galaxies', ax=ax[-1])
ax[-1].set_xticks(ticks=jnp.arange(len(masses)), labels=bound_m, rotation=30)
ax[-1].set_yticks(ticks=jnp.arange(len(redshifts)), labels=bound_z)
ax[-1].set_xlabel(r'$\log M_* / M_\odot$')
ax[-1].set_ylabel(r'Redshift $z$')
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()
plt.savefig('data.png', dpi=300, bbox_inches='tight')