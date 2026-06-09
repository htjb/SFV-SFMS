"""Plot the main sequence scatter posteriors."""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astroemu.network import mlp
from burstiness.prior import prior as prior_dict
from astroemu.normalisation import log_base_10, standardise
from astroemu.serialisation import load as astroemu_load

plt.rcParams.update({"font.size": 10})

kernel = sys.argv[1]  # "ext-reg" or "sho"
nsamples = 10000
nparams = 2500
ndims = 5 if kernel == "ext-reg" else 3
nlive = ndims * 250

with open(
    "observations/intrinsic_scatter_per_timescale_errors.yaml",
) as f:
    obs = yaml.safe_load(f)

redshifts = [3.5, 4.5, 5.5, 6.5, 7.5]
masses = ["M_A", "M_B", "M_C", "M_D"]
red = ["z3_4", "z4_5", "z5_6", "z6_7", "z7_8"]
mass_bins = {
    "M_A": "8.0_8.5", "M_B": "8.5_9.0",
    "M_C": "9.0_9.5", "M_D": "9.5_10.0",
}
latex_m = [
    r"A: $[8, 8.5]$",
    r"B: $(8.5, 9]$",
    r"C: $(9, 9.5]$",
    r"D: $(9.5, 10]$",
]

# Fixed timescales (Myr labels, Gyr values)
times_myr = [10, 20, 30, 40, 50, 100]
times_gyr = np.array([t / 1000 for t in times_myr])
plot_times_index = [0, 2, 5]  # 10, 30, 100 Myr
skips = {"M_C_z7_8", "M_D_z6_7", "M_D_z7_8"}


def make_emulate_batch(emulator_dir: str):
    """Load astroemu emulator and return a vmapped batch prediction fn."""
    loaded = astroemu_load(os.path.join(emulator_dir, "emulator.astroemu"))
    net_params = loaded["params"]
    activation = loaded["hyperparams"]["act"]
    pipeline = loaded["train_pipeline"]

    std_pipe = next(p for p in pipeline if isinstance(p, standardise))
    log_pipe = next(p for p in pipeline if isinstance(p, log_base_10))
    log_input = log_pipe.log_all_params

    p_mean = jnp.array(std_pipe.params_mean)
    p_std = jnp.array(std_pipe.params_std)
    x_mean = jnp.array(std_pipe.x_mean)
    x_std = jnp.array(std_pipe.x_std)
    y_mean = jnp.array(std_pipe.y_mean)
    y_std = jnp.array(std_pipe.y_std)
    log_t = jnp.array(np.log10(times_gyr))
    n_t = len(times_gyr)

    def emulate_one(log_params: jnp.ndarray) -> jnp.ndarray:
        if log_input:
            pn = (log_params - p_mean) / p_std
        else:
            pn = (10.0**log_params - p_mean) / p_std
        tn = (log_t - x_mean) / x_std
        inputs = jnp.concatenate(
            [tn[:, None], jnp.tile(pn[None, :], (n_t, 1))], axis=-1
        )
        raw = mlp(net_params, inputs, act=activation).squeeze(-1)
        return 10.0 ** (raw * y_std + y_mean)

    return jax.vmap(emulate_one)


pnames = (
    ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]
    if kernel == "ext-reg"
    else ["s0", "tau0", "Q"]
)
rng = np.random.default_rng(0)
log10_prior_lo = np.array([np.log10(prior_dict[p]["min"]) for p in pnames])
log10_prior_hi = np.array([np.log10(prior_dict[p]["max"]) for p in pnames])
prior_log10_samples = (
    rng.uniform(size=(200, ndims))
    * (log10_prior_hi - log10_prior_lo)
    + log10_prior_lo
)

median_sigmas, onesigmas, twosigmas = [], [], []
datasets, data_low_errors, data_high_errors = [], [], []
all_signals = []
prior_signals = []
emulator_error = 0.0  # updated per bin

for m in masses:
    median_sigmas_m, onesigmas_m, twosigmas_m = [], [], []
    datasets_m, data_low_errors_m, data_high_errors_m = [], [], []
    all_signals_m = []
    prior_signals_m = []

    for z_val, z in zip(redshifts, red):
        target = f"{m}_{z}"
        if target in skips:
            nan_row = np.full(len(times_myr), np.nan)
            median_sigmas_m.append(nan_row)
            onesigmas_m.append(np.array([nan_row, nan_row]))
            twosigmas_m.append(np.array([nan_row, nan_row]))
            datasets_m.append(np.full(len(times_myr), np.nan))
            data_low_errors_m.append(np.zeros(len(times_myr)))
            data_high_errors_m.append(np.zeros(len(times_myr)))
            all_signals_m.append(np.full((1, len(times_myr)), np.nan))
            prior_signals_m.append(np.full((1, len(times_myr)), np.nan))
            continue
        mb = mass_bins[m]
        chains_dir = (
            f"lbi/blackjax_chains_astroemu-{kernel}-sfr_theory_mean"
            f"-{nparams}-{nsamples}_z_{z_val}_mass_bin_{mb}_nlive_{nlive}"
        )
        emulator_dir = (
            f"lbi/fit_burstiness_project/burstiness-emulator-astroemu-"
            f"{kernel}-sfr_theory_mean-{nparams}-{nsamples}"
            f"_z_{z_val}_mass_bin_{mb}/"
        )
        npz_path = f"{chains_dir}/{target}/posterior.npz"

        d = np.array(obs[target]["data"])
        error_low = np.array(obs[target]["low_error"])
        error_high = np.array(obs[target]["high_error"])
        datasets_m.append(d)
        data_low_errors_m.append(error_low)
        data_high_errors_m.append(error_high)

        try:
            samples_log10 = np.load(npz_path)["samples_log10"]
            emulator_error = float(
                np.loadtxt(os.path.join(emulator_dir, "mean_error.txt"))
            ) / 100.0
            emulate_batch = make_emulate_batch(emulator_dir)
            sigmas = np.array(
                emulate_batch(jnp.array(samples_log10))
            )  # (N, n_times)
        except FileNotFoundError:
            nan_row = np.full(len(times_myr), np.nan)
            median_sigmas_m.append(nan_row)
            onesigmas_m.append(np.array([nan_row, nan_row]))
            twosigmas_m.append(np.array([nan_row, nan_row]))
            all_signals_m.append(np.full((1, len(times_myr)), np.nan))
            prior_signals_m.append(np.full((1, len(times_myr)), np.nan))
            continue

        median_sigmas_m.append(np.mean(sigmas, axis=0))
        onesigmas_m.append(np.percentile(sigmas, [16, 84], axis=0))
        twosigmas_m.append(np.percentile(sigmas, [2.5, 97.5], axis=0))
        all_signals_m.append(sigmas)
        prior_signals_m.append(
            np.array(emulate_batch(jnp.array(prior_log10_samples)))
        )

    median_sigmas.append(median_sigmas_m)
    onesigmas.append(onesigmas_m)
    twosigmas.append(twosigmas_m)
    datasets.append(datasets_m)
    data_low_errors.append(data_low_errors_m)
    data_high_errors.append(data_high_errors_m)
    all_signals.append(all_signals_m)
    prior_signals.append(prior_signals_m)

median_sigmas = np.array(median_sigmas)
onesigmas = np.array(onesigmas)
twosigmas = np.array(twosigmas)
datasets = np.array(datasets)
data_low_errors = np.array(data_low_errors)
data_high_errors = np.array(data_high_errors)

#cols = plt.cm.magma(np.linspace(0.1, 0.7, 7))
#colours = [cols[2], cols[3], "dodgerblue", cols[6]]
colours = [plt.get_cmap('Reds_r')(0.3)] * 4


def plot_error_bars(ax, j, pti):
    """Compute and plot error bars for a given mass bin and time index."""
    med = median_sigmas[j, :, pti]
    d = datasets[j, :, pti]
    el = data_low_errors[j, :, pti]
    eh = data_high_errors[j, :, pti]
    sigma = el + (eh - el) * (med > d)
    for k in range(len(sigma)):
        if sigma[k] / d[k] < 0.03:
            sigma[k] = 0.03 * d[k]
        sigma[k] = np.sqrt(sigma[k] ** 2 + (emulator_error * d[k]) ** 2)
    ax.errorbar(
        redshifts, d, yerr=sigma,
        ls="", marker="o", markersize=4,
        c=colours[j], label="Data",
        capsize=3, ecolor="black", markeredgecolor="black",
    )


# --- Figure 1: individual sample traces ---
fig, axes = plt.subplots(4, 3, figsize=(6.3, 6.3), sharex=True, sharey=True)
for i, ax_col in enumerate(axes.T):
    pti = plot_times_index[i]
    for j in range(len(masses)):
        # all_signals[j] = list of (N, n_times) arrays, one per redshift
        for s in range(len(prior_log10_samples)):
            y_prior = [
                prior_signals[j][z_idx][s, pti]
                if prior_signals[j][z_idx].shape[0] > 1
                else np.nan
                for z_idx in range(len(red))
            ]
            ax_col[j].plot(
                redshifts, y_prior,
                color="gray", alpha=0.05, zorder=-20, lw=0.5,
            )
        n_samples = min(
            s.shape[0] for s in all_signals[j] if s.shape[0] > 1
        )
        for s in range(n_samples):
            y = [
                all_signals[j][z_idx][s, pti]
                if all_signals[j][z_idx].shape[0] > 1
                else np.nan
                for z_idx in range(len(red))
            ]
            ax_col[j].plot(
                redshifts, y,
                color=colours[j], alpha=0.02, zorder=-10,
            )
        ax_col[j].plot(
            redshifts, median_sigmas[j, :, pti],
            c="k", ls="--", label=r"Median $\sigma_\mathrm{MS}$",
        )
        plot_error_bars(ax_col[j], j, pti)
        ax_col[j].set_ylim([0, 0.6])

[axes[0, i].set_title(f"{times_myr[pti]} Myr")
 for i, pti in enumerate(plot_times_index)]
fig.supxlabel(r"$z$", y=0.02, fontsize=12)
[axes[i, 0].set_ylabel(latex_m[i], fontsize=10) for i in range(len(masses))]
fig.supylabel(r"$\sigma_\mathrm{MS}$", x=0.04, fontsize=12)
for i in range(axes.shape[0]):
    if i > 0:
        axes[i, 0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
[axes[-1, i].set_xticks([4, 5, 6, 7]) for i in range(len(plot_times_index))]
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f"lbi/{kernel}_sigma.png", dpi=300)
plt.close()

# --- Figure 2: credible interval ribbons ---
fig, axes = plt.subplots(4, 3, figsize=(5, 5), sharex=True, sharey=True)
for i, ax_col in enumerate(axes.T):
    pti = plot_times_index[i]
    for j in range(len(masses)):
        for s in range(len(prior_log10_samples)):
            y_prior = [
                prior_signals[j][z_idx][s, pti]
                if prior_signals[j][z_idx].shape[0] > 1
                else np.nan
                for z_idx in range(len(red))
            ]
            ax_col[j].plot(
                redshifts, y_prior,
                color="gray", alpha=0.2, zorder=-20, lw=0.5,
            )
        ax_col[j].fill_between(
            redshifts, twosigmas[j, :, 0, pti], twosigmas[j, :, 1, pti],
            color=colours[j], alpha=0.2,
        )
        ax_col[j].fill_between(
            redshifts, onesigmas[j, :, 0, pti], onesigmas[j, :, 1, pti],
            color=colours[j], alpha=0.8,
        )
        ax_col[j].plot(
            redshifts, median_sigmas[j, :, pti],
            c="k", ls="--", label=r"Median $\sigma_\mathrm{MS}$",
        )
        plot_error_bars(ax_col[j], j, pti)
        ax_col[j].set_ylim([0, 0.6])

[axes[0, i].set_title(f"{times_myr[pti]} Myr")
 for i, pti in enumerate(plot_times_index)]
fig.supxlabel(r"Redshift $z$")
[axes[i, 0].set_ylabel(latex_m[i]) for i in range(len(masses))]
fig.supylabel(r"$\sigma_\mathrm{MS}$", x=0.04, fontsize=12)
for i in range(axes.shape[0]):
    if i > 0:
        axes[i, 0].set_yticks([0.1, 0.3, 0.5])
[axes[-1, i].set_xticks([4, 5, 6, 7]) for i in range(len(plot_times_index))]
fig.suptitle('SHO' if kernel == 'sho' else 'ExtReg', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f"lbi/{kernel}_sigmas_confidence.png", dpi=300)
plt.close()
