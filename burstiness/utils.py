"""Utility functions for burstiness project."""

import numpy as np
from jax import numpy as jnp

def calc_extreg_psd(tau: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the extended regulator model power spectral density.

    Args:
        tau: Timescales at which to compute the PSD (ntau,)
        theta: Model parameters (5,):
            [sigma, tau_eq, tau_in, sigma_gmc, tau_gmc]

    Returns:
        PSD values at the given timescales (ntau,)
    """
    sigma, tau_eq, tau_in, sigma_gmc, tau_gmc = theta
    sreg = sigma**2 / (
        1
        + ((2 * jnp.pi * tau_in) ** 2 + (2 * np.pi * tau_eq) ** 2)
        * (1 / tau) ** 2
        + (2 * jnp.pi * tau_in) ** 2
        * (2 * jnp.pi * tau_eq) ** 2
        * (1 / tau) ** 4
    )
    sgmc = sigma_gmc**2 / (1 + (2 * jnp.pi * tau_gmc) ** 2 * (1 / tau) ** 2)
    return sreg + sgmc


def calc_reg_psd(tau: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the regulator model power spectral density."""
    sigma, tau_eq, tau_in, _, _ = theta
    sreg = sigma**2 / (
        1
        + ((2 * np.pi * tau_in) ** 2 + (2 * np.pi * tau_eq) ** 2)
        * (1 / tau) ** 2
        + (2 * np.pi * tau_in) ** 2
        * (2 * np.pi * tau_eq) ** 2
        * (1 / tau) ** 4
    )
    return sreg


def calc_gmc_psd(tau: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the GMC model power spectral density."""
    _, _, _, sigma_gmc, tau_gmc = theta
    sgmc = sigma_gmc**2 / (1 + (2 * np.pi * tau_gmc) ** 2 * (1 / tau) ** 2)
    return sgmc


def calc_sho_psd(tau: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the SHO model power spectral density."""
    s0, tau_0, Q = theta
    w0 = 1 / tau_0
    sho = (
        np.sqrt(2 / np.pi)
        * (s0 * w0**4)
        / (((1 / tau) ** 2 - w0**2) ** 2 + (1 / tau) ** 2 * w0**2 / Q**2)
    )
    return sho


def integrate(
    sfh: jnp.ndarray, tarr: jnp.ndarray, lim: float | int, tuniv: float | int
) -> jnp.ndarray:
    """Integrate the SFH from (tuniv - lim) to tuniv.

    Args:
        sfh: Star formation history array (nsamp, ntarr)
        tarr: Time array corresponding to the SFH (ntarr,)
        lim: Integration limit in Myr
        tuniv: Age of the universe at the given redshift in Myr

    Returns:
        Integrated SFH values (nsamp,)
    """
    tbegin = tuniv - lim
    mask = tarr > tbegin
    sfh_cut = sfh[mask]
    tarr_cut = tarr[mask]
    return jnp.trapezoid(sfh_cut, tarr_cut)