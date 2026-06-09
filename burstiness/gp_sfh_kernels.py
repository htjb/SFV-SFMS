"""Kernel functions for Gaussian Processes star formation histories."""

import jax.numpy as jnp

# --------------------- kernels -------------------------------------


def regulator_model_kernel(
    delta_t: jnp.ndarray, kernel_params: list
) -> jnp.ndarray:
    """Regulator model kernel, with three parameters.

    Args:
        delta_t (jnp.array): Time difference array.
        kernel_params (list): List of kernel parameters
            [sigma, tau_eq, tau_in].
    """
    sigma, tau_eq, tau_in = kernel_params

    tau = jnp.abs(delta_t)

    c_reg = (
        sigma**2
        / (tau_in - tau_eq)
        * (tau_in * jnp.exp(-tau / tau_in) - tau_eq * jnp.exp(-tau / tau_eq))
    )

    return c_reg


def extended_regulator_model_kernel_paramlist(
    delta_t: jnp.ndarray, kernel_params: list
) -> jnp.ndarray:
    """Extended regulator model kernel, with five parameters.

    Args:
        delta_t (jnp.array): Time difference array.
        kernel_params (list): List of kernel parameters [sigma, tau_eq, tau_in,
        sigma_gmc, tau_gmc].

    Returns:
        jnp.ndarray: Kernel values.
    """
    sigma, tau_eq, tau_in, sigma_gmc, tau_gmc = kernel_params

    tau = jnp.abs(delta_t)

    c_reg = (
        sigma**2
        / (tau_in - tau_eq)
        * (tau_in * jnp.exp(-tau / tau_in) - tau_eq * jnp.exp(-tau / tau_eq))
    )

    c_gmc = sigma_gmc**2 * jnp.exp(-tau / tau_gmc)

    kernel_val = c_reg + c_gmc
    return kernel_val


def sho_kernel(
    delta_t: jnp.ndarray, kernel_params: list = [10, 1, 2]
) -> jnp.ndarray:
    """SHO kernel implementation.

    Based on: https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F;
    Figure 1 and Eqn 20.

    Args:
        delta_t (jnp.ndarray): Time difference array.
        kernel_params (list): List of kernel parameters [s0, tau0, Q].

    Returns:
        jnp.ndarray: Kernel values.
    """
    s0, tau0, Q = kernel_params
    factor = jnp.sqrt(jnp.abs(1 - (4 * Q**2) ** (-1)))
    omega0 = 1 / tau0
    a = s0 * omega0 * Q

    tau = jnp.abs(delta_t)
    qpo = (
        a
        * jnp.exp(-omega0 * tau / (2 * Q))
        * (
            jnp.cos(factor * omega0 * tau)
            + 1 / (2 * factor * Q) * jnp.sin(factor * omega0 * tau)
        )
    )

    return qpo
