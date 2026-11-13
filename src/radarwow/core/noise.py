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


def signal_mask(ds: xr.Dataset, margin_db: float = 3.0, snr_min: float = 3.0, sqi_min: float = 0.4,
                snr_trust_db: float | None = None) -> np.ndarray:
    """Boolean (azimuth, range) mask of gates with real echo.

    Uses every available quality moment (SNR > snr_min AND SQI/NCP > sqi_min): one alone can be
    biased (e.g. the ARM CSAPR2 SNR field marks 98 % of gates as signal while NCP marks ~42 %).
    Where the source's SNR is trustworthy, a gate with SNR >= ``snr_trust_db`` counts as signal whatever
    its SQI. The reader sets this per format (sweep attribute ``radarwow_snr_trust_db``; ODIM: 15 dB):
    FMI's SQI is low in rain (median 0.5-0.8 at 20-40 dBZ, SNR 37-50 dB), and the SQI test alone discarded
    10-15 % of those rain gates (34 % at 10-20 dBZ). It is off by default because some SNR fields are
    unreliable (ARM CSAPR2, above). Without SNR or SQI, falls back to the reflectivity noise floor.
    """
    if snr_trust_db is None:
        snr_trust_db = ds.attrs.get("radarwow_snr_trust_db")
    z = ds["DBZH"].values.astype("float64")
    has_snr, has_sqi = "SNRH" in ds, "SQIH" in ds
    if has_snr or has_sqi:
        snr = np.asarray(ds["SNRH"].values) if has_snr else None
        ok = (snr > snr_min) if has_snr else np.ones(z.shape, bool)
        if has_sqi:
            sqi_ok = np.asarray(ds["SQIH"].values) > sqi_min
            ok &= (sqi_ok | (snr >= snr_trust_db)) if (has_snr and snr_trust_db is not None) else sqi_ok
        return ok & np.isfinite(z)
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
