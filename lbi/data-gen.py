"""Generate emulator training data for burstiness analysis."""

import os
import sys
from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from burstiness.gp_sfh_jax import get_tarr, get_tuniv, sample_kernel
from burstiness.gp_sfh_kernels import (
    extended_regulator_model_kernel_paramlist,
    sho_kernel,
)
from burstiness.mean_function import sfr_theoretical_mean
from burstiness.prior import prior
from burstiness.utils import integrate

key = jax.random.PRNGKey(42)

# CLI: kernel zval mass_bin [seed_mass] [nparams] [ngals]
# e.g. python lbi/data-gen.py ext-reg 3.5 8_8.5 1e7 2500 10000
kernel_arg = sys.argv[1]
zval_arg = float(sys.argv[2])
mb_arg = sys.argv[3]  # e.g. "8_8.5"
seed_arg = float(sys.argv[4]) if len(sys.argv) > 4 else 1e7
nparams = int(sys.argv[5]) if len(sys.argv) > 5 else 2500
ngals = int(sys.argv[6]) if len(sys.argv) > 6 else 10000

mb_lo, mb_hi = float(mb_arg.split("_")[0]), float(mb_arg.split("_")[1])

config = {
    "kernel": kernel_arg,
    "zval": zval_arg,
    "mass_bin": [mb_lo, mb_hi],
    "seed_mass": seed_arg,
    "tarr_res": 500,
}

config["base_dir"] = (
    "training-data/"
    + config["kernel"]
    + "-sfr_theory_mean-"
    + str(nparams)
    + "-"
    + str(ngals)
    + "/"
)

os.makedirs(config["base_dir"], exist_ok=True)

now = datetime.now()
current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current DateTime:", current_datetime)


def sample_from_prior(kernel_type: str, n_samples: int) -> np.ndarray:
    """Sample parameters from prior distributions."""
    if kernel_type == "sho":
        params = np.column_stack(
            [
                10
                ** np.random.uniform(
                    np.log10(prior["s0"]["min"]),
                    np.log10(prior["s0"]["max"]),
                    n_samples,
                ),
                10
                ** np.random.uniform(
                    np.log10(prior["tau0"]["min"]),
                    np.log10(prior["tau0"]["max"]),
                    n_samples,
                ),
                10
                ** np.random.uniform(
                    np.log10(prior["Q"]["min"]),
                    np.log10(prior["Q"]["max"]),
                    n_samples,
                ),
            ]
        )
    elif kernel_type == "ext-reg":
        params = np.column_stack(
            [
                10
                ** np.random.uniform(
                    np.log10(prior["sig_reg"]["min"]),
                    np.log10(prior["sig_reg"]["max"]),
                    n_samples,
                ),
                10
                ** np.random.uniform(
                    np.log10(prior["tau_eq"]["min"]),
                    np.log10(prior["tau_eq"]["max"]),
                    n_samples,
                ),
                10
                ** np.random.uniform(
                    np.log10(prior["tau_in"]["min"]),
                    np.log10(prior["tau_in"]["max"]),
                    n_samples,
                ),
                10
                ** np.random.uniform(
                    np.log10(prior["sig_dyn"]["min"]),
                    np.log10(prior["sig_dyn"]["max"]),
                    n_samples,
                ),
                10
                ** np.random.uniform(
                    np.log10(prior["tau_dyn"]["min"]),
                    np.log10(prior["tau_dyn"]["max"]),
                    n_samples,
                ),
            ]
        )
    return params


params = sample_from_prior(config["kernel"], nparams)

integrate = jax.vmap(integrate, in_axes=(0, None, None, None))

tuniv = get_tuniv(config["zval"])

if config["kernel"] == "sho-fixed-Q" or config["kernel"] == "sho":
    kernel = sho_kernel
elif config["kernel"] == "ext-reg":
    kernel = extended_regulator_model_kernel_paramlist

times = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

print("-" * 50)

tarr = get_tarr(n_tarr=config["tarr_res"], zval=config["zval"])
print(
    "t resolution number of samples",
    config["tarr_res"],
    "Resolution (Gyr):",
    tarr[1] - tarr[0],
)
with h5py.File(
    config["base_dir"]
    + "emulation_data_z_"
    + str(config["zval"])
    + "_mass_bin_"
    + str(config["mass_bin"][0])
    + "_"
    + str(config["mass_bin"][1])
    + ".h5",
    "w",
) as f:
    f.create_group("masses")
    f.create_group("sfr_subsamples")
    all_masses = []
    all_stds = []
    evaluated_params = []
    pbar = tqdm(range(len(params)), desc="Generating data")
    skipped_count = 0
    for i in pbar:
        split_key, key = jax.random.split(key)

        samples = sample_kernel(
            split_key,
            kernel,
            nsamp=ngals,
            n_tarr=config["tarr_res"],
            zval=config["zval"],
            kernel_params=params[i],
            mean_func=sfr_theoretical_mean,
            mean_function_kwargs={
                "mhalo_seed": config["seed_mass"],
            },
        )
        # this is the log10 SFRs so convert to SFRs
        galaxies = 10**samples  # in Msun/yr

        masses = jax.vmap(jnp.trapezoid, in_axes=(0, None))(
            galaxies, tarr * 1e9
        )  # in Msun

        masses = jnp.log10(masses)
        mask = (masses > config["mass_bin"][0]) & (
            masses < config["mass_bin"][1]
        )
        n_in_bin = int(jnp.sum(mask))

        if n_in_bin < 30:
            skipped_count += 1
            pbar.set_postfix(
                {"skipped": str(skipped_count) + "/" + str(i + 1)}
            )
            continue

        evaluated_params.append(params[i])

        stds_massbin = []
        for t in range(len(times)):
            # average SFR over times[t] Gyr — keep full fixed-
            # size array for JIT
            sigma = jnp.log10(
                integrate(galaxies, tarr * 1e9, times[t] * 1e9, tuniv * 1e9)
                / (times[t] * 1e9)
            )
            # scatter over in-bin galaxies only, using fixed-size
            #  masked mean/var
            mean_s = jnp.sum(jnp.where(mask, sigma, 0.0)) / n_in_bin
            var_s = (
                jnp.sum(jnp.where(mask, (sigma - mean_s) ** 2, 0.0)) / n_in_bin
            )
            stds_massbin.append(jnp.sqrt(var_s))

        stds_massbin = np.array(stds_massbin)
        all_stds.append(stds_massbin)
        all_masses.extend(masses[mask].tolist())

        f["masses"].create_dataset("set" + str(i), data=masses[mask])
        galaxies_in_bin = galaxies[mask]
        n_subsample = min(100, n_in_bin)
        f["sfr_subsamples"].create_dataset(
            "set" + str(i), data=galaxies_in_bin[:n_subsample]
        )

    f.create_dataset("scatters", data=all_stds)
    f.create_dataset("parameters", data=evaluated_params)
    f.attrs["times"] = times
    if config["kernel"] == "sho":
        param_names = np.array(list(prior.keys()), dtype="S")[:3]
    elif config["kernel"] == "ext-reg":
        param_names = np.array(list(prior.keys()), dtype="S")[3:]
    f.attrs["parameter_names"] = param_names
    f.attrs["mass_bin"] = np.array(config["mass_bin"])
    f.attrs["zval"] = np.array(config["zval"])
    f.attrs["generation_time"] = np.array(current_datetime, dtype="S")
    f.attrs["seed_mass"] = np.array(config["seed_mass"])
    f.attrs["tarr_res"] = np.array(config["tarr_res"])
    f.attrs["kernel"] = np.array(config["kernel"], dtype="S")
all_masses = np.array(all_masses)
# all_sfrs = np.array(all_sfrs)
print("Overall mass range:", np.min(all_masses), np.max(all_masses))
print("Average mass:", np.mean(all_masses))

"""plt.hist(all_masses, bins=50)
plt.xlabel("log10 Stellar Mass (Msun)")
plt.ylabel("Number of Galaxies")
plt.title("Distribution of Stellar Masses in Generated Data")
plt.show()"""
