"""Plot control parameter posterior distributions for a single control run."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from anesthetic import Samples

from burstiness.prior import prior as prior_dict

plt.rcParams.update({"font.size": 10})

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

if kernel == "ext-reg":
    pnames = ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]
    latex_names = [
        r"$\log \sigma_\mathrm{Reg}$",
        r"$\log \tau_\mathrm{eq}$",
        r"$\log \tau_\mathrm{in}$",
        r"$\log \sigma_\mathrm{Dyn}$",
        r"$\log \tau_\mathrm{Dyn}$",
    ]
else:
    pnames = ["s0", "tau0", "Q"]
    latex_names = [r"$\log s_0$", r"$\log \tau_0$", r"$\log Q$"]

log10_prior_lo = np.array(
    [np.log10(prior_dict[p]["min"]) for p in pnames]
)
log10_prior_hi = np.array(
    [np.log10(prior_dict[p]["max"]) for p in pnames]
)

# Truth parameters (physical space) → convert to log10
test_params_phys = np.loadtxt(emulator_dir + "test_params.txt")
truth_phys = test_params_phys[control_indices[control]]
truth_log10 = np.log10(truth_phys)

# Posterior samples (already log10)
samples_log10 = np.load(path)["samples_log10"]

# Prior samples
rng = np.random.default_rng(0)
prior_samples = (
    rng.uniform(size=(2000, ndims))
    * (log10_prior_hi - log10_prior_lo)
    + log10_prior_lo
)

prior_s = Samples(data=prior_samples, columns=pnames, labels=latex_names)
post_s = Samples(data=samples_log10, columns=pnames, labels=latex_names)

ax = prior_s.plot_2d(pnames, label="Prior", color="gray", alpha=0.5)
post_s.plot_2d(ax, label="Posterior", color="steelblue")

truth_dict = {p: truth_log10[i] for i, p in enumerate(pnames)}
ax.axlines(truth_dict, color="r", lw=1.5, ls="--", label="Truth")

plt.savefig(
    f"{chains_dir}/{data_tag}/{control}_parameter_posteriors.png",
    dpi=300, bbox_inches="tight",
)
plt.close("all")
