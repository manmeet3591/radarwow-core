"""Signal vs noise: some sources censor noise gates (NaN), others store them as values.

Receiver noise appears in reflectivity as Z_noise(r) = N0 + 20 log10(r): constant power,
rising with range. Uncensored noise therefore looks exactly like an interference spike,
and it pollutes texture, KDP and rhoHV statistics. `signal_mask` returns gates that are
real echo, using SNR if available, otherwise an estimated noise floor.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def noise_floor(z: np.ndarray, r_m: np.ndarray, pct: float = 5.0) -> tuple[np.ndarray, float, bool]:
    """Estimate the reflectivity noise curve N0 + 20 log10(r).

    Returns (curve over range, N0, censored?). ``censored`` is True when the source already
    blanks noise (many NaN, no gates near a 20 log r floor).
    """
    r_km = np.maximum(r_m / 1e3, 0.1)
    finite = np.isfinite(z)
    if finite.mean() < 0.6:
        return np.full(r_km.shape, -np.inf), -np.inf, True
    zc = z - 20 * np.log10(r_km)[None, :]
    n0 = float(np.nanpercentile(zc, pct))
    return n0 + 20 * np.log10(r_km), n0, False


def snr_estimate(ds: xr.Dataset) -> np.ndarray:
    """Signal-to-noise ratio (dB) per gate.

    Uses the file's SNR field when present (a biased SNR field, as on ARM CSAPR2, is caught by the
    SQI test in `signal_mask`, which callers combine with this). Otherwise SNR ~ Z - (N0 + 20 log10 r)
    from the noise curve; if the source censors noise (e.g. NEXRAD), N0 is taken from the weakest
    echoes kept (1st percentile, the censoring threshold), so the estimate is a lower bound.
    """
    if "SNRH" in ds:
        return ds["SNRH"].values.astype("float64")
    z = ds["DBZH"].values.astype("float64")
    r_km = np.maximum(ds["range"].values / 1e3, 0.1)
    curve, _, censored = noise_floor(z, ds["range"].values)
    if censored:
        zc = z - 20 * np.log10(r_km)[None, :]
        curve = float(np.nanpercentile(zc, 1.0)) + 20 * np.log10(r_km)
    return z - curve[None, :]


def signal_mask(ds: xr.Dataset, margin_db: float = 3.0, snr_min: float = 3.0, sqi_min: float = 0.4) -> np.ndarray:
    """Boolean (azimuth, range) mask of gates with real echo.

    Uses every available quality moment (SNR > snr_min AND SQI/NCP > sqi_min): one alone can be
    biased (e.g. the ARM CSAPR2 SNR field marks 98 % of gates as signal while NCP marks ~42 %).
    Without either, falls back to the reflectivity noise floor.
    """
    masks = []
    if "SNRH" in ds:
        masks.append(np.asarray(ds["SNRH"].values > snr_min))
    if "SQIH" in ds:
        masks.append(np.asarray(ds["SQIH"].values > sqi_min))
    z = ds["DBZH"].values.astype("float64")
    if masks:
        return np.logical_and.reduce(masks) & np.isfinite(z)
    curve, _, censored = noise_floor(z, ds["range"].values)
    if censored:
        return np.isfinite(z)
    return np.isfinite(z) & (z > curve[None, :] + margin_db)


def mask_nexrad_codes(da: xr.DataArray) -> tuple[xr.DataArray, list]:
    """NEXRAD Level II stores 'below threshold' (code 0) and 'range folded' (code 1) as the two
    smallest decodable values (e.g. DBZH -33 / -32.5, ZDR -8, RHOHV 0.2017). Set them to NaN."""
    x = da.values
    vals = np.unique(x[np.isfinite(x)])
    if vals.size < 3:
        return da, []
    step = float(np.min(np.diff(vals)))  # quantisation step = 1 / scale
    codes = [float(vals[0]), float(vals[0] + step)]
    out = da.where(~np.isclose(da, codes[0]) & ~np.isclose(da, codes[1], atol=step * 0.1))
    out.attrs = {**da.attrs, "radarwow_codes_removed": codes}
    return out, codes
