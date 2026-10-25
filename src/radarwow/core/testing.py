"""Synthetic rays and sweeps with known truth, for unit tests and notebook checks."""

from __future__ import annotations

import numpy as np
import xarray as xr


def cell_kdp(r_km, centers=(40.0,), widths=(6.0,), peaks=(3.0,)):
    """KDP (deg/km) as a sum of Gaussian rain cells."""
    k = np.zeros_like(r_km, dtype=float)
    for c, w, p in zip(centers, widths, peaks):
        k += p * np.exp(-0.5 * ((r_km - c) / w) ** 2)
    return k


def synthetic_ray(n=600, dr_m=250.0, offset=60.0, noise=3.0, delta_peak=0.0, wrap=False, seed=0, **cells):
    """Return (r_km, psi_measured, phidp_true, kdp_true).

    psi = offset + 2*int(KDP) + delta (backscatter bump at the cell core) + Gaussian noise,
    optionally wrapped to (-180, 180].
    """
    rng = np.random.default_rng(seed)
    r_km = np.arange(n) * dr_m / 1e3
    kdp = cell_kdp(r_km, **cells)
    phidp = offset + 2 * np.cumsum(kdp) * dr_m / 1e3
    delta = 0.0
    if delta_peak:
        c = cells.get("centers", (40.0,))[0]
        delta = delta_peak * np.exp(-0.5 * ((r_km - c) / 1.0) ** 2)
    psi = phidp + delta + rng.normal(0, noise, n)
    if wrap:
        psi = (psi + 180) % 360 - 180
    return r_km, psi, phidp, kdp


def synthetic_sweep(n_az=360, n=600, dr_m=250.0, seed=0, clutter_patch=True, spike_az=None) -> xr.Dataset:
    """Sweep with a rain cell, an optional clutter patch and an optional interference spike."""
    rng = np.random.default_rng(seed)
    az = np.arange(n_az) + 0.5
    r = np.arange(n) * dr_m + dr_m / 2
    A, R = np.meshgrid(az, r / 1e3, indexing="ij")
    z = 45 * np.exp(-(((A - 90) / 25) ** 2 + ((R - 60) / 20) ** 2)) + rng.normal(0, 1.0, A.shape)
    z = np.where(z > 5, z, np.nan)
    zdr = np.where(np.isfinite(z), 0.02 * np.nan_to_num(z) + rng.normal(0, 0.2, A.shape), np.nan)
    rho = np.where(np.isfinite(z), 0.99 + rng.normal(0, 0.005, A.shape), np.nan)
    kdp = np.where(np.isfinite(z), 1e-3 * 10 ** (0.06 * np.nan_to_num(z)), 0.0)
    phi = 40 + 2 * np.cumsum(kdp, axis=1) * dr_m / 1e3 + rng.normal(0, 3, A.shape)
    phi = np.where(np.isfinite(z), phi, np.nan)
    truth = np.zeros(A.shape, dtype=bool)
    if clutter_patch:
        m = (A > 200) & (A < 215) & (R > 8) & (R < 20)
        z[m] = rng.uniform(20, 65, m.sum())
        zdr[m] = rng.normal(0, 3, m.sum())
        phi[m] = rng.uniform(-180, 180, m.sum())
        rho[m] = rng.uniform(0.4, 0.9, m.sum())
        truth[m] = True
    if spike_az is not None:
        i = int(spike_az)
        z[i, :] = 5 + 20 * np.log10(R[i, :] + 0.1) + rng.normal(0, 1, n)
        rho[i, :] = rng.uniform(0.1, 0.5, n)
        phi[i, :] = rng.uniform(-180, 180, n)
        zdr[i, :] = rng.normal(0, 4, n)
    ds = xr.Dataset({"DBZH": (("azimuth", "range"), z), "ZDR": (("azimuth", "range"), zdr),
                     "PHIDP": (("azimuth", "range"), phi), "RHOHV": (("azimuth", "range"), rho),
                     "CLUTTER_TRUTH": (("azimuth", "range"), truth)},
                    coords={"azimuth": az, "range": r, "elevation": ("azimuth", np.full(n_az, 0.5))})
    return ds
