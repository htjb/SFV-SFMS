"""Generate example data for the illustrative plots."""

import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from burstiness.gp_sfh_jax import get_tarr, get_tuniv, sample_kernel
from burstiness.gp_sfh_kernels import (
    extended_regulator_model_kernel_paramlist,
    sho_kernel,
)
from burstiness.mean_function import sfr_theoretical_mean
from burstiness.utils import calc_extreg_psd, calc_sho_psd, integrate

plt.rcParams.update({"font.size": 8})

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

key = jax.random.PRNGKey(21)


# np.random.seed(12)
def plot_extreg_acf(
    tau: jnp.ndarray, theta: jnp.ndarray, dyn: bool = False
) -> jnp.ndarray:
    """Calculate the acf for the extended regulator model kernel."""
    sigma, tau_eq, tau_in, sigma_gmc, tau_gmc = theta
    tau = np.abs(tau)
    if dyn:
        c_gmc = sigma_gmc**2 * np.exp(-tau / tau_gmc)
        return c_gmc
    else:
        c_reg = (
            sigma**2
            / (tau_in - tau_eq)
            * (tau_in * np.exp(-tau / tau_in) - tau_eq * np.exp(-tau / tau_eq))
        )

        c_gmc = sigma_gmc**2 * np.exp(-tau / tau_gmc)

        kernel_val = c_reg + c_gmc
        return kernel_val


def plot_sho_acf(tau: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Calculate the acf for the SHO kernel."""
    tau = np.abs(tau)
    s0, tau0, Q = theta
    omega0 = 1 / tau0
    eta = np.sqrt(np.abs(1 - 1 / (4 * Q**2)))
    return (
        s0
        * omega0
        * Q
        * np.exp(-omega0 * tau / (2 * Q))
        * (
            np.cos(eta * omega0 * tau)
            + 1 / (2 * eta * Q) * np.sin(eta * omega0 * tau)
        )
    )


kernel = "ext-reg"
ndims = 5 if kernel == "ext-reg" else 3

ngals = 2500

if kernel == "sho":
    # S0, tau0, Q
    params = np.vstack([0.002, 0.03, 2.0], dtype="float32").T
elif kernel == "ext-reg":
    # sig_reg, tau_eq, tau_in, sig_dyn, tau_dyn
    params = np.vstack([0.8, 0.5, 0.1, 0.4, 0.01]).T

integrate = jax.vmap(integrate, in_axes=(0, None, None, None))

zval = 3

tuniv = get_tuniv(zval)
if kernel == "ext-reg":
    kernel_func = extended_regulator_model_kernel_paramlist
else:
    kernel_func = sho_kernel
times = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

tarr_res = 1000

tarr = get_tarr(
    n_tarr=tarr_res, zval=zval,
)
delta_t = tarr[:, None] - tarr[None, :]

psd_plot_tarr = np.linspace(0.001, 10, 1000)
sfr_plot_tarr = tarr #np.linspace(0.001, tuniv, 200)

if kernel == "ext-reg":
    plot_psd = calc_extreg_psd
    plot_acf = plot_extreg_acf
elif kernel == "sho":
    plot_psd = calc_sho_psd
    plot_acf = plot_sho_acf


fig = plt.figure(figsize=(6.3, 4.0))
gs = GridSpec(2, 6, figure=fig)
axes = [
    fig.add_subplot(gs[0, :2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:]),
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]

for i in tqdm(range(len(params))):
    split_key, key = jax.random.split(key)
    samples = sample_kernel(
        split_key,
        kernel_func,
        nsamp=ngals,
        n_tarr=tarr_res,
        zval=zval,
        kernel_params=params[i],
        mean_func=sfr_theoretical_mean,
        mean_function_kwargs={
            "mhalo_seed": 1e7,
        }
    )

    mean_sfr, halo_mass, stellar_mass = (
        sfr_theoretical_mean(tarr, mhalo_seed=1e7)
    )
    #mean_sfr  = mean_sfr * fb

    galaxies = 10**samples
    stds, means = [], []
    for t in range(len(times)):
        sigma = jnp.log10(
            integrate(galaxies, tarr * 1e9, times[t] * 1e9, tuniv * 1e9)
            / (times[t] * 1e9)
        )
        parts = axes[3].violinplot(
            sigma,
            positions=[times[t]],
            widths=0.01,
            showmeans=True,
            showmedians=False,
            showextrema=False,
            quantiles=[0.16, 0.84],
        )
        for pc in parts["bodies"]:
            pc.set_color("k")

        parts["cmeans"].set_color("k")
        parts["cmeans"].set_linewidth(1)
        parts["cquantiles"].set_color("k")
        parts["cquantiles"].set_linewidth(1)

        stds.append(jnp.std(sigma))
        means.append(jnp.mean(sigma))

    stds = np.array(stds)
    axes[4].plot(
        np.array(times),
        stds,
        color="k",
        marker="o",
    )

    axes[0].plot(
        psd_plot_tarr,
        plot_psd(psd_plot_tarr, params[i]),
        color="k",
        label="Total" if kernel == "ext-reg" else None,
    )
    if kernel == "ext-reg":
        reg = [params[i][0], params[i][1], params[i][2], 1e-6, 1e6]
        axes[0].plot(
            psd_plot_tarr,
            plot_psd(psd_plot_tarr, reg),
            color="k",
            ls=":",
            label="Regulator",
        )
        dyn = [1e-6, 1e6, 1e6, params[i][3], params[i][4]]
        axes[0].plot(
            psd_plot_tarr,
            plot_psd(psd_plot_tarr, dyn),
            color="k",
            ls="--",
            label="Dynamical",
        )

    #axes[1].plot(delta_t, plot_acf(delta_t, params[i]), color="k")
    #if kernel == "ext-reg":
        # axes[1].plot(delta_t, plot_acf(delta_t, reg), color="k", ls=":")
        # axes[1].plot(
        #     delta_t, plot_acf(delta_t, dyn, dyn=True), color="k", ls="--"
        # )
    acf = plot_acf(delta_t, params[i])
    im = axes[1].imshow(
        np.log10(acf),
        extent=(tarr[0], tarr[-1], tarr[0], tarr[-1]),
        origin="lower",
        aspect="auto",
        cmap=plt.get_cmap("Greys"),
    )
    plt.colorbar(im, ax=axes[1], label=r"ACF [dex$^2$]", 
                 fraction=0.3, pad=0.04)
    axes[1].set(xlabel="Time [Gyr]", xscale="log", yscale="log", ylabel="Time [Gyr]")

    for j in range(5):
        sfr = jnp.interp(sfr_plot_tarr, tarr, 10**samples[j])
        axes[2].plot(sfr_plot_tarr, sfr, c="k", alpha=j / 5) 
    #[axes[2].plot(tarr, 10**samples[j], c="k", alpha=j / 5) for j in range(5)]
    axes[2].plot(tarr, 10**(mean_sfr / np.log(10)), c="w", ls="--", label="Mean function")

axes[0].set(xlabel="Timescale [Gyr]", yscale="log", xscale="log")
axes[0].set(ylabel=r"PSD [dex$^2$ Gyr]")
if i == 0 and kernel == "ext-reg":
    axes[0].legend(fontsize=6, loc="lower left")
axes[0].set(ylim=(1e-8, 1e1))

axes[2].set(ylabel=r"$\mathrm{SFR}$ [$M_\odot$ yr$^{-1}$]")
#axes[2].set_xscale("log")
z_ticks = [3, 4, 6, 15]
t_ticks = [cosmo.age(z).value for z in z_ticks]
t_ticks, z_ticks = zip(*[
    (t, z) for t, z in zip(t_ticks, z_ticks) if t <= tuniv
])
axes[2].set_xlim(min(t_ticks), tuniv)
axes[2].set_xticks(t_ticks)
axes[2].set_xticklabels([str(z) for z in z_ticks])
axes[2].set_xlabel("Redshift")
# axes[2].set(ylim=(-6, 5))
axes[4].set_xlabel(r"Timescale [Gyr]", fontdict={"size": 10})
axes[3].set_xlabel(r"Timescale [Gyr]", fontdict={"size": 10})
axes[4].set_ylabel(r"$\sigma_\mathrm{MS}$ [dex]", fontdict={"size": 10})
axes[3].set_ylabel(
    r"$\log {\rm SFR}_{\rm t_{avg}}$ [M$_\odot$ yr$^{-1}$]",
    fontdict={"size": 10},
)

if kernel == "ext-reg":
    axes[0].set_ylim(bottom=1e-5)

for i in range(len(axes)):
    if i != 1:
        axes[i].grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(wspace=10)

plt.savefig(
    "example-data-" + kernel + ".png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
plt.close()
