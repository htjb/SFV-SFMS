"""Plot the PSD posterior for a single control run."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours
from matplotlib.patches import FancyArrowPatch

from burstiness.prior import prior as prior_dict
from burstiness.utils import calc_extreg_psd, calc_sho_psd

plt.rcParams.update({"font.size": 8})

kernel = sys.argv[1]        # "sho" or "ext-reg"
z = float(sys.argv[2])      # e.g. 3.5
mass_bin = sys.argv[3]      # e.g. "8_8.5"
control = sys.argv[4] if len(sys.argv) > 4 else "control"
nsamples = 10000
nparams = 2500
ndims = 5 if kernel == "ext-reg" else 3
nlive = ndims * 250

mass_bin_map = {
    "8_8.5": ("8.0_8.5", "M_A"),
    "8.5_9": ("8.5_9.0", "M_B"),
    "9_9.5": ("9.0_9.5", "M_C"),
    "9.5_10": ("9.5_10.0", "M_D"),
}
z_tag_map = {3.5: "z3_4", 4.5: "z4_5", 5.5: "z5_6", 6.5: "z6_7", 7.5: "z7_8"}
control_indices = {"control": 0, "control2": 61, "control3": 4}

mb, m_tag = mass_bin_map[mass_bin]
z_tag = z_tag_map[z]
data_tag = f"{m_tag}_{z_tag}"

emulator_dir = (
    f"lbi/fit_burstiness_project/burstiness-emulator-astroemu-"
    f"{kernel}-sfr_theory_mean-{nparams}-{nsamples}_z_{z}_mass_bin_{mb}/"
)
chains_dir = (
    f"lbi/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
    f"-{nparams}-{nsamples}_z_{z}_mass_bin_{mb}_nlive_{nlive}"
)
path = f"{chains_dir}/{data_tag}/{control}/posterior.npz"

calc_psd = calc_sho_psd if kernel == "sho" else calc_extreg_psd

if kernel == "sho":
    pnames = ["s0", "tau0", "Q"]
else:
    pnames = ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]

log10_prior_lo = np.array(
    [np.log10(prior_dict[p]["min"]) for p in pnames]
)
log10_prior_hi = np.array(
    [np.log10(prior_dict[p]["max"]) for p in pnames]
)

# Truth parameters
test_params_phys = np.loadtxt(emulator_dir + "test_params.txt")
truth_phys = test_params_phys[control_indices[control]]

# Posterior samples → physical space for calc_psd
theta = 10 ** np.load(path)["samples_log10"]

# Prior samples
rng = np.random.default_rng(0)
prior_samples = 10 ** (
    rng.uniform(size=(250, ndims))
    * (log10_prior_hi - log10_prior_lo)
    + log10_prior_lo
)

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
t_obs = cosmo.age(z).to("Gyr").value
t_form = cosmo.age(30).to("Gyr").value
t_evolve = t_obs - t_form  # total evolution time

tau = np.logspace(-3, np.log10(t_evolve), 1000)

mean_psd = 10 ** np.median(np.log10([calc_psd(tau, th) for th in theta]), axis=0)

fig, ax = plt.subplots(1, 1, figsize=(3.15, 3))

plot_contours(
    calc_psd, tau, theta,
    ax=ax, colors=plt.get_cmap(
                "Reds_r"
            ),
            contour_line_levels=[1, 2], fineness=1,
            alpha=0.6, ny=500 if kernel == "sho" else 100,
)
for th in prior_samples:
    ax.plot(tau, calc_psd(tau, th), alpha=0.4, c="gray", lw=0.8, zorder=-10)

ax.plot(tau, mean_psd, c="k", lw=1.5, label="Posterior Median", zorder=10)
ax.plot(
    tau, calc_psd(tau, truth_phys),
    c="red", lw=1.5, ls="--", label="Truth", zorder=10,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=1e-7)
ax.set_xlabel(r"Timescale [Gyr]")
ax.set_ylabel(r"PSD [dex$^2$ Gyr]")
ax.axvline(0.01, ls=":", color='k')
ax.axvline(0.1, ls=":", color='k')
ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig(
    f"{chains_dir}/{data_tag}/{control}_psd_posterior_{kernel}.png",
    dpi=300, bbox_inches="tight",
)
plt.close()
