"""Mean functions for the PSD GP."""

import warnings

import jax
import jax.numpy as jnp
from astropy.cosmology import FlatLambdaCDM
from jax.experimental.ode import odeint

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

fb = cosmo.Ob0 / cosmo.Om0  # cosmic baryon fraction

# Precompute z(t) and H(z) grids for fast JAX interpolation
_H0 = cosmo.H(0).value
_z_grid = jnp.linspace(0.0, 30.0, 2000)              # ascending z, covers all redshifts used
_t_grid = jnp.array(cosmo.age(_z_grid).value) * 1e9  # years, ascending in t, descending in z
_H_grid = jnp.array([cosmo.H(float(z)).value for z in _z_grid])

@jax.jit
def sfr_theoretical_mean(
    tarr: jnp.ndarray,
    mhalo_seed: float = 1e6,
    epsilon: float = 0.1,
    m0: float = 1e12,
    alpha: float = 0.6,
    beta: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Mass and redshift dependent mean function for the SFH GP.

    Solves the halo mass ODE dM/dt = accretion_rate(M, z(t)),
    where the accretion rate from Rodriguez-Puebla+2016 depends on the
    current halo mass at each timestep.

    Args:
        tarr (jnp.ndarray): Time array in Gyr.
        mhalo_seed (float): Seed halo mass in solar masses at t=tarr[0].
            Defaults to 1e6.
        epsilon (float): Star formation efficiency. Defaults to 0.1.
        m0 (float): Characteristic mass scale in solar masses. Defaults to 1e12.
        alpha (float): Power-law slope for low masses. Defaults to 0.6.
        beta (float): Power-law slope for high masses. Defaults to 0.5.

    Returns:
        jnp.ndarray: Mean SFR in M☉/yr at each time in tarr.
    """
    tarr_yr = jnp.array(tarr) * 1e9  # Convert Gyr to years

    def _z_at_t(t_yr):
        """Interpolate redshift from time in years using precomputed grid.

        _t_grid is descending (early universe = high z = small t),
        so we flip both arrays for jnp.interp which needs ascending xp.
        """
        return jnp.interp(t_yr, jnp.flip(_t_grid), jnp.flip(_z_grid))

    def _H_at_z(z):
        """Interpolate H(z) in km/s/Mpc using precomputed grid."""
        return jnp.interp(z, _z_grid, _H_grid)

    def accretion_rate(M, t_yr):
        """Accretion rate dM/dt from Rodriguez-Puebla+2016.

        Signature is (state, t) for odeint.
        """
        z = _z_at_t(t_yr)
        a = 1.0 / (1.0 + z)
        gamma = 1.0 + 0.329 * a - 0.206 * a ** 2
        log10C = 2.730 - 1.828 * a + 0.654 * a ** 2
        H_ratio = _H_at_z(z) / _H0
        return 10 ** log10C * (M / m0) ** gamma * H_ratio

    # odeint expects f(y, t) where y and return value have the same shape
    halo_mass = odeint(
        lambda M, t: jnp.atleast_1d(accretion_rate(M[0], t)),
        jnp.array([mhalo_seed]),
        tarr_yr,
        rtol=1e-6,
        atol=1e-9,
    )[:, 0]  # shape (n_tarr,)

    # vectorise accretion_rate over time for sfr computation
    dMdt = jax.vmap(accretion_rate)(halo_mass, tarr_yr)
    sfr = (
        2 * epsilon
        / ((halo_mass / m0) ** (-alpha) + (halo_mass / m0) ** beta)
        * dMdt
        * fb
    )
    stellar_mass = jnp.trapezoid(sfr, tarr_yr)
    log_sfr = jnp.log(sfr)
    return log_sfr, halo_mass, stellar_mass