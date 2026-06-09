"""Inference using the astroemu emulator and blackjax nested slice sampling."""

import os
import sys

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from anesthetic import NestedSamples, Samples
from astroemu.network import mlp
from astroemu.normalisation import log_base_10, standardise
from astroemu.serialisation import load as astroemu_load
from blackjax.ns import nss, utils
from tqdm import tqdm

from burstiness.prior import prior as prior_dict

# ---------------------------------------------------------------------------
# CLI args  (mirror tuning-astroemu.py)
# ---------------------------------------------------------------------------
kernel = sys.argv[1]   # "ext-reg" or "sho"
mass_bin = sys.argv[2]  # "8_8.5", "8.5_9", etc.
zval = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
nparams = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
ngals = int(sys.argv[5]) if len(sys.argv) > 5 else 10000

mapping = {"8_8.5": "M_A", "8.5_9": "M_B", "9_9.5": "M_C", "9.5_10": "M_D"}
data_bin = mapping[mass_bin]

# Convert mass_bin to format like "8.0_8.5" for consistency with data files
mass_bin = (
    str(float(mass_bin.split("_")[0]))
    + "_"
    + str(float(mass_bin.split("_")[1]))
)

ndims = 3 if kernel == "sho" else 5
num_live = ndims * 250
num_inner_steps = ndims * 5   # recommended: multiple of ndims
dlogZ_tolerance = 0.01        # stop when <1% evidence remains in live points
max_steps = 100_000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir = (
    "training-data/"
    + kernel
    + "-sfr_theory_mean-"
    + str(nparams)
    + "-"
    + str(ngals)
    + "/"
)
data_file = (
    data_dir
    + "emulation_data_z_"
    + str(zval)
    + "_mass_bin_"
    + mass_bin
    + ".h5"
)
emulator_dir = (
    "lbi/fit_burstiness_project/burstiness-emulator-astroemu-"
    + kernel
    + f"-sfr_theory_mean-{nparams}-{ngals}_z_{zval}_mass_bin_{mass_bin}/"
)
base_dir = (
    "lbi/chains/blackjax_chains_astroemu-"
    + kernel
    + f"-sfr_theory_mean-{nparams}-{ngals}_z_{zval}_mass_bin_{mass_bin}"
    + f"_nlive_{num_live}/"
)
os.makedirs(base_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load emulator
# ---------------------------------------------------------------------------
loaded = astroemu_load(os.path.join(emulator_dir, "emulator.astroemu"))
net_params = loaded["params"]
activation = loaded["hyperparams"]["act"]
pipeline = loaded["train_pipeline"]

# Extract normalisation components from the saved pipeline
std_pipe = next(p for p in pipeline if isinstance(p, standardise))
log_pipe = next(p for p in pipeline if isinstance(p, log_base_10))
log_input = log_pipe.log_all_params   # True if params were log10'd during training

emulator_error = (
    float(np.loadtxt(os.path.join(emulator_dir, "mean_error.txt"))) / 100.0
)

# Timescales from the HDF5 training data file
with h5py.File(data_file, "r") as f:
    times = np.array(f.attrs["times"])   # (n_times,) in Gyr
log_times = jnp.array(np.log10(times))
n_times = len(times)

# Test set saved by tuning-astroemu.py — used for control runs
test_stds = np.loadtxt(os.path.join(emulator_dir, "test_stds.txt"))     # (n_test, n_times)
test_params_raw = np.loadtxt(os.path.join(emulator_dir, "test_params.txt"))  # (n_test, ndims)

# ---------------------------------------------------------------------------
# Emulate function (JAX-JIT-able, operates in log10 parameter space)
# ---------------------------------------------------------------------------
_params_mean = jnp.array(std_pipe.params_mean)
_params_std = jnp.array(std_pipe.params_std)
_x_mean = jnp.array(std_pipe.x_mean)
_x_std = jnp.array(std_pipe.x_std)
_y_mean = jnp.array(std_pipe.y_mean)
_y_std = jnp.array(std_pipe.y_std)


@jax.jit
def emulate(log_params: jnp.ndarray) -> jnp.ndarray:
    """Predict scatter at all timescales for a single set of log10 params.

    The network was trained on z-scored inputs. Timescales are always in
    log10 space; kernel parameters are either in log10 space (if log_input)
    or physical space (if not log_input) for the standardise step.

    Args:
        log_params: Log10 kernel parameters, shape (ndims,).

    Returns:
        Predicted scatter in dex, shape (n_times,).
    """
    if log_input:
        params_normed = (log_params - _params_mean) / _params_std
    else:
        # Pipeline was trained on physical-space params — convert back
        params_normed = (10.0**log_params - _params_mean) / _params_std

    times_normed = (log_times - _x_mean) / _x_std

    # Build (n_times, 1 + ndims) tiled inputs and run the network
    tiled_params = jnp.tile(params_normed[None, :], (n_times, 1))
    inputs = jnp.concatenate([times_normed[:, None], tiled_params], axis=-1)
    raw_preds = mlp(net_params, inputs, act=activation).squeeze(-1)   # (n_times,)

    # Denormalise: undo z-score, then undo log10 (log_all_y=True in pipeline)
    log_preds = raw_preds * _y_std + _y_mean
    return 10.0**log_preds   # dex


# ---------------------------------------------------------------------------
# Prior: log-uniform on all kernel parameters (uniform in log10 space)
# ---------------------------------------------------------------------------
if kernel == "sho":
    prior_bounds_phys = np.array([
        [prior_dict["s0"]["min"],    prior_dict["s0"]["max"]],
        [prior_dict["tau0"]["min"],  prior_dict["tau0"]["max"]],
        [prior_dict["Q"]["min"],     prior_dict["Q"]["max"]],
    ])
    param_names = ["s0", "tau0", "Q"]
    param_labels = [r"$\log S_0$", r"$\log \tau_0$", r"$\log Q$"]
elif kernel == "ext-reg":
    prior_bounds_phys = np.array([
        [prior_dict["sig_reg"]["min"], prior_dict["sig_reg"]["max"]],
        [prior_dict["tau_eq"]["min"],  prior_dict["tau_eq"]["max"]],
        [prior_dict["tau_in"]["min"],  prior_dict["tau_in"]["max"]],
        [prior_dict["sig_dyn"]["min"], prior_dict["sig_dyn"]["max"]],
        [prior_dict["tau_dyn"]["min"], prior_dict["tau_dyn"]["max"]],
    ])
    param_names = ["sig_reg", "tau_eq", "tau_in", "sig_dyn", "tau_dyn"]
    param_labels = [
        r"$\log \sigma_\mathrm{reg}$",
        r"$\log \tau_\mathrm{eq}$",
        r"$\log \tau_\mathrm{in}$",
        r"$\log \sigma_\mathrm{dyn}$",
        r"$\log \tau_\mathrm{dyn}$",
    ]

log_prior_bounds = np.log10(prior_bounds_phys)   # (ndims, 2) in log10 space
_log_prior_lo = jnp.array(log_prior_bounds[:, 0])
_log_prior_width = jnp.array(log_prior_bounds[:, 1] - log_prior_bounds[:, 0])


def logprior_fn(log_params: jnp.ndarray) -> jnp.ndarray:
    """Log prior: uniform on log10 parameter bounds (= log-uniform in physical space).

    Args:
        log_params: Single particle position, shape (ndims,).

    Returns:
        Scalar log-prior probability.
    """
    return jnp.sum(
        jax.scipy.stats.uniform.logpdf(log_params, _log_prior_lo, _log_prior_width)
    )


# ---------------------------------------------------------------------------
# Observation data
# ---------------------------------------------------------------------------
with open("observations/intrinsic_scatter_per_timescale_errors.yaml") as f:
    obs_data = yaml.safe_load(f)

# Add control targets from the emulator test set
control_indices = {"control": 0, "control2": 61, "control3": 4}
for ctrl_name, idx in control_indices.items():
    if idx < len(test_stds):
        # test_stds.txt is in dex (predict() fully denormalises including log10)
        d_ctrl = test_stds[idx]
        obs_data[ctrl_name] = {
            "data": d_ctrl.tolist(),
            "low_error": (emulator_error * d_ctrl).tolist(),
            "high_error": (emulator_error * d_ctrl).tolist(),
        }

# Construct the single observational target from CLI args, e.g. "M_A_z_3_4"
z_lo = int(zval - 0.5)
z_hi = int(zval + 0.5)
data_tag = f"{data_bin}_z{z_lo}_{z_hi}"

targets = list(control_indices.keys()) + [data_tag]

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
rng_key = jax.random.PRNGKey(42)

for target in targets:
    # Controls are nested under the data_tag folder so they stay with their emulator
    if target in control_indices:
        fitdir = os.path.join(base_dir, data_tag, target)
    else:
        fitdir = os.path.join(base_dir, data_tag)
    os.makedirs(fitdir, exist_ok=True)
    print(f"\n--- {target} ---")

    d = jnp.array(obs_data[target]["data"])
    error_low = jnp.array(obs_data[target]["low_error"])
    error_high = jnp.array(obs_data[target]["high_error"])

    # Factory avoids closing over loop variables by value
    def make_loglikelihood(d, error_low, error_high):
        def loglikelihood_fn(log_params: jnp.ndarray) -> jnp.ndarray:
            """Gaussian likelihood with asymmetric errors and emulator noise floor.

            Args:
                log_params: Single particle, shape (ndims,).

            Returns:
                Scalar log-likelihood.
            """
            emulated = emulate(log_params)
            # Asymmetric observational errors
            sigma = jnp.where(emulated > d, error_high, error_low)
            # Floor at 3% to avoid numerical issues
            sigma = jnp.where(sigma / d < 0.03, 0.03 * d, sigma)
            total_var = sigma**2 + (emulator_error * d)**2
            return jnp.sum(
                -0.5 * jnp.log(2.0 * jnp.pi * total_var)
                - 0.5 * (d - emulated)**2 / total_var
            )
        return loglikelihood_fn

    loglikelihood_fn = make_loglikelihood(d, error_low, error_high)

    # Build sampler and JIT the step
    algo = nss.as_top_level_api(logprior_fn, loglikelihood_fn, num_inner_steps,
                                num_delete=num_live//4)
    step_fn = jax.jit(algo.step)

    # Initialise live points uniformly from the prior
    rng_key, init_key = jax.random.split(rng_key)
    init_positions = (
        jax.random.uniform(init_key, (num_live, ndims))
        * _log_prior_width + _log_prior_lo
    )
    state = algo.init(init_positions)

    # NS loop — stop when live-point evidence fraction < dlogZ_tolerance
    dead = []
    pbar = tqdm(range(max_steps), desc=f"NS [{target}]")
    for _ in pbar:
        rng_key, step_key = jax.random.split(rng_key)
        state, info = step_fn(step_key, state)
        dead.append(info)

        logZ_dead = state.integrator.logZ
        logZ_live = state.integrator.logZ_live
        logZ_total = jnp.logaddexp(logZ_dead, logZ_live)
        remaining = float(jnp.exp(logZ_live - logZ_total))

        pbar.set_postfix({
            "logZ": f"{float(logZ_total):.2f}",
            "rem%": f"{100 * remaining:.1f}",
        })

        if remaining < dlogZ_tolerance:
            break

    # Combine all dead particles with the final live points
    all_dead = utils.finalise(state, dead, update_info=False)

    logZ_final = float(jnp.logaddexp(state.integrator.logZ, state.integrator.logZ_live))
    ns_ess = float(utils.ess(rng_key, all_dead))
    print(f"logZ = {logZ_final:.3f}  |  ESS = {ns_ess:.0f}  |  steps = {len(dead)}")

    # Draw posterior samples by importance resampling
    rng_key, sample_key = jax.random.split(rng_key)
    posterior = utils.sample(sample_key, all_dead, shape=int(ns_ess))
    samples_log10 = np.array(posterior.position)   # (n_samples, ndims), log10 space

    # Save
    np.savez(
        os.path.join(fitdir, "posterior.npz"),
        samples_log10=samples_log10,
        samples_phys=10**samples_log10,
        logZ=logZ_final,
        ess=ns_ess,
        param_names=param_names,
        times=np.array(times),
    )
    np.savetxt(os.path.join(fitdir, "logZ.txt"), [logZ_final])

    # ---------------------------------------------------------------------------
    # Corner plot with anesthetic
    # ---------------------------------------------------------------------------
    logL_birth = np.array(all_dead.particles.loglikelihood_birth)
    logL_birth = np.where(np.isfinite(logL_birth), logL_birth, -1e30)
    ns_samples = NestedSamples(
        data=np.array(all_dead.particles.position),
        logL=np.array(all_dead.particles.loglikelihood),
        logL_birth=logL_birth,
        columns=param_names,
        labels=param_labels,
    )

    rng_prior = np.random.default_rng(0)
    prior_data = (
        rng_prior.uniform(size=(2000, ndims))
        * np.array(_log_prior_width)
        + np.array(_log_prior_lo)
    )
    prior_samples = Samples(
        data=prior_data, columns=param_names, labels=param_labels
    )
    ax = prior_samples.plot_2d(
        param_names, label="Prior", color="gray", alpha=0.5
    )
    ns_samples.plot_2d(ax, label="Posterior", color="steelblue")

    if target in control_indices:
        ctrl_idx = control_indices[target]
        if ctrl_idx < len(test_params_raw):
            truth_log10 = {
                p: np.log10(test_params_raw[ctrl_idx, i])
                for i, p in enumerate(param_names)
            }
            ax.axlines(truth_log10, color="r", lw=1.5, ls="--", label="Truth")

    ax.iloc[-1, 0].legend()
    plt.savefig(os.path.join(fitdir, "posterior_corner.png"), dpi=150)
    plt.close("all")

    # -----------------------------------------------------------------------
    # Predictive check: emulated scatter vs observed
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    # Draw predicted scatters for a subset of posterior samples
    for i in range(int(ns_ess)):
        pred = np.array(emulate(jnp.array(samples_log10[i])))
        ax.plot(times, pred, color="steelblue", alpha=0.05, lw=0.8)

    d_np = np.array(d)
    el_np = np.array(error_low)
    eh_np = np.array(error_high)
    ax.errorbar(
        times, d_np, yerr=[el_np, eh_np],
        fmt="ko", ms=4, capsize=3, label="Data", zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Timescale (Gyr)")
    ax.set_ylabel("Scatter (dex)")
    ax.set_title(f"{target} — predictive check")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fitdir, "predictive_check.png"), dpi=150)
    plt.close()

    print(f"Results saved to {fitdir}")
