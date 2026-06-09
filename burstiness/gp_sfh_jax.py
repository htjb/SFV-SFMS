"""SFH Gaussian Process utilities using JAX."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from astropy.cosmology import FlatLambdaCDM

from burstiness.mean_function import constant_mean

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

_tarr_cache = {}


def get_tuniv(zval: float | int = 0.1) -> float:
    """Get the age of the universe at a given redshift.

    Args:
        zval (float | int, optional): Redshift value. Defaults to 0.1.

    Returns:
        float: Age of the universe at the given redshift in Gyr.
    """
    t_univ = cosmo.age(zval).value
    return t_univ


def get_tarr(
    zval: int | float = 0.1,
    n_tarr: int = 1000,
) -> jnp.ndarray:
    """Get the time array.

    Args:
        zval (int | float, optional): Redshift value. Defaults to 0.1.
        n_tarr (int, optional): Number of time array points. Defaults to 1000.
        integration_period (int | float, optional): Integration period in Gyr.
            If provided, the time array will span from
            (tuniv - integration_period)
            to tuniv. Defaults to None.

    Returns:
        jnp.ndarray: Time array in Gyr.
    """
    key = (zval, n_tarr)
    if key not in _tarr_cache:
        t_univ = cosmo.age(zval).value
        tbegin = cosmo.age(30).value
        _tarr_cache[key] = jnp.linspace(tbegin, t_univ, n_tarr)
    return _tarr_cache[key]


def get_covariance_matrix(
    kernel: Callable,
    n_tarr: int = 1000,
    zval: int | float = 0.1,
    **kwargs: dict,
) -> jnp.ndarray:
    """Get the covariance matrix for the SFH kernel.

    Args:
        kernel (Callable): Kernel function to use for the GP.
        n_tarr (int, optional): Number of time array points. Defaults to 1000.
        zval (int | float, optional): Redshift value. Defaults to 0.1.
        **kwargs (dict): Additional keyword arguments for the kernel.

    Returns:
        jnp.ndarray: Covariance matrix.
    """
    tarr = get_tarr(n_tarr=n_tarr, zval=zval)
    delta_t = tarr[:, None] - tarr[None, :]
    delta_t_flat = jax.numpy.array(delta_t.flatten())

    kernel = partial(kernel, **kwargs)
    kernel = jax.vmap(jax.jit(kernel), in_axes=(0))
    cov_flat = kernel(delta_t_flat)
    cov_matrix = cov_flat.reshape(delta_t.shape)
    return cov_matrix


def sample_kernel(
    key: jnp.ndarray,
    kernel: Callable,
    nsamp: int = 100,
    n_tarr: int = 1000,
    zval: float | int = 0.1,
    mean_func: Callable = constant_mean,
    mean_function_kwargs: dict = {},
    **kwargs: dict,
) -> jnp.ndarray:
    """Sample the SFH kernel using a Gaussian Process.

    Args:
        key (jnp.ndarray): JAX random key.
        kernel (Callable): Kernel function to use for the GP.
        nsamp (int, optional): Number of samples to draw. Defaults to 100.
        n_tarr (int, optional): Number of time array points. Defaults to 1000.
        zval (float | int, optional): Redshift value. Defaults to 0.1.
        mean_func (Callable, optional): Mean function for the GP.
            Defaults to constant_mean.
        mean_function_kwargs (dict, optional): Keyword arguments
             for the mean function. Defaults to {}.
        **kwargs (dict): Additional keyword arguments for the kernel.

    Returns:
        jnp.ndarray: Samples from the GP representing log SFRs.
    """
    # tarr is in Gyrs but it just sets the sampling of the GP and
    # how correlated
    # points are so units don't matter here
    # units of the SFH come from the mean function which is set to 1 Msun/yr
    tarr = get_tarr(n_tarr=n_tarr, zval=zval)  # in Gyr
    mean_array, halo_mass, stellar_mass = mean_func(
        tarr, **mean_function_kwargs
    )

    covariance_matrix = get_covariance_matrix(
        kernel, n_tarr=n_tarr, zval=zval, **kwargs
    )

    # np.random.seed(random_seed)
    z = jax.random.normal(key, (nsamp, len(tarr)))
    L = jnp.linalg.cholesky(covariance_matrix + 1e-3 * jnp.eye(len(tarr)))

    samples = mean_array + z @ L.T

    # returns ln SFR
    return samples / jnp.log(10)  # convert to log10 SFR
