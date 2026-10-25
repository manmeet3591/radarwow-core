"""Canonical radar moment names and per-source aliases.

Every radar-wow algorithm reads and writes the ODIM/CfRadial2 short names below.
Sources name the same quantity differently (NEXRAD, Py-ART/CMAC long names, ARM,
IMD IRIS-netCDF, MXPol), so `standardize` renames the *raw* field of each kind to
its canonical name. Already-processed copies (e.g. ARM CMAC ``corrected_*``) are
left untouched and reported by `radarwow.core.diagnose`.
"""

from __future__ import annotations

import xarray as xr

#: canonical name -> (description, units)
CANONICAL = {
    "DBZH": ("horizontal reflectivity (clutter-filtered if the source filters)", "dBZ"),
    "DBTH": ("total (unfiltered) horizontal reflectivity", "dBZ"),
    "ZDR": ("differential reflectivity", "dB"),
    "PHIDP": ("measured differential phase (propagation + backscatter + system offset)", "degrees"),
    "RHOHV": ("co-polar correlation coefficient", "1"),
    "KDP": ("specific differential phase", "degrees/km"),
    "VRADH": ("radial velocity", "m/s"),
    "WRADH": ("spectrum width", "m/s"),
    "SNRH": ("signal-to-noise ratio", "dB"),
    "SQIH": ("signal quality index / normalized coherent power", "1"),
    "HCLASS": ("vendor hydrometeor classification", "1"),
}

#: canonical -> source names, in priority order (raw before processed)
ALIASES = {
    "DBZH": ["DBZH", "reflectivity", "DBZ", "reflectivity_horizontal", "Z", "dBZ", "DBZHC"],
    "DBTH": ["DBTH", "TH", "T", "total_power", "unfiltered_reflectivity"],
    "ZDR": ["ZDR", "differential_reflectivity", "diff_reflectivity"],
    "PHIDP": ["PHIDP", "UPHIDP", "differential_phase", "diff_phase", "dp_phase_shift"],
    "RHOHV": ["RHOHV", "cross_correlation_ratio", "copol_correlation_coeff", "copol_coeff", "RHO"],
    "KDP": ["KDP", "specific_differential_phase"],
    "VRADH": ["VRADH", "velocity", "VEL", "V", "mean_doppler_velocity"],
    "WRADH": ["WRADH", "spectrum_width", "WIDTH", "W"],
    "SNRH": ["SNRH", "signal_to_noise_ratio", "SNR", "SNRHC"],
    "SQIH": ["SQIH", "normalized_coherent_power", "NCP", "norm_coherent_power"],
    "HCLASS": ["HCLASS"],
}

PROCESSED_HINTS = ("corrected_", "filtered_", "attenuation_corrected", "proc_", "recalculated_", "unf_")

#: CF / CfRadial standard names; metadata wins over variable names (e.g. one ARM file stores
#: KDP in a variable called ``diff_phase``).
CF_NAMES = {
    "DBZH": ("equivalent_reflectivity_factor",),
    "ZDR": ("log_differential_reflectivity_hv",),
    "PHIDP": ("differential_phase_hv",),
    "KDP": ("specific_differential_phase_hv",),
    "RHOHV": ("cross_correlation_ratio_hv",),
    "VRADH": ("radial_velocity_of_scatterers_away_from_instrument",),
    "WRADH": ("doppler_spectrum_width",),
    "SNRH": ("signal_to_noise_ratio",),
}
_CF_TO_CANON = {cf: c for c, names in CF_NAMES.items() for cf in names}


def cf_kind(da) -> str | None:
    """Canonical moment implied by a variable's standard_name or long_name, if any."""
    for key in ("standard_name", "long_name"):
        val = str(da.attrs.get(key, "")).strip().lower()
        if val in _CF_TO_CANON:
            return _CF_TO_CANON[val]
    return None


def standardize(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
    """Rename the first raw alias of each moment to its canonical name.

    Returns the renamed dataset and the mapping {canonical: original}.
    Variables whose names start with a processed-product prefix are never renamed.
    """
    mapping: dict[str, str] = {}
    rename: dict[str, str] = {}
    present = list(ds.data_vars)
    kinds = {n: cf_kind(ds[n]) for n in present}
    taken: set[str] = set()
    for canon, names in ALIASES.items():
        if canon in present:
            mapping[canon] = canon
            taken.add(canon)
            continue
        raw = [n for n in present if not n.startswith(PROCESSED_HINTS) and n not in taken]
        # 1) metadata says it is this moment (alias order breaks ties)
        by_meta = [n for n in raw if kinds[n] == canon]
        by_meta.sort(key=lambda n: names.index(n) if n in names else len(names))
        # 2) name matches and metadata does not contradict it
        by_name = [n for n in names if n in raw and kinds[n] in (None, canon)]
        pick = (by_meta or by_name or [None])[0]
        if pick:
            rename[pick] = canon
            mapping[canon] = pick
            taken.add(pick)
    out = ds.rename(rename)
    for canon in mapping:
        out[canon].attrs.setdefault("long_name", CANONICAL[canon][0])
        out[canon].attrs["radarwow_source_name"] = mapping[canon]
    return out, mapping


def has(ds: xr.Dataset, *moments: str) -> bool:
    return all(m in ds.data_vars for m in moments)


def missing(ds: xr.Dataset, moments) -> list[str]:
    return [m for m in moments if m not in ds.data_vars]


def phidp_direction(ds: xr.Dataset, min_gates: int = 30) -> tuple[int, float]:
    """Sign convention of PHIDP: +1 if it increases with range in rain (standard), -1 if it decreases.

    Fits the slope of the unfolded PHIDP vs range on rain gates (rhoHV > 0.95, Z > 30 dBZ) per ray
    and returns (sign, fraction of rays agreeing). Some radars (e.g. ARM CSAPR2 here) store the
    opposite sign.
    """
    import numpy as np

    if "PHIDP" not in ds:
        return 1, 0.0
    phi = ds["PHIDP"].values.astype(float)
    good = np.isfinite(phi)
    if "RHOHV" in ds:
        good &= ds["RHOHV"].values > 0.95
    if "DBZH" in ds:
        good &= ds["DBZH"].values > 30
    r = ds["range"].values / 1e3
    slopes = []
    for i in range(phi.shape[0]):
        idx = np.flatnonzero(good[i])
        if idx.size >= min_gates and r[idx[-1]] - r[idx[0]] > 5:
            p = np.rad2deg(np.unwrap(np.deg2rad(phi[i, idx])))
            slopes.append(np.polyfit(r[idx], p, 1)[0])
    if len(slopes) < 5:
        return 1, 0.0
    slopes = np.asarray(slopes)
    neg = float(np.mean(slopes < 0))
    return (-1, neg) if neg > 0.7 else (1, 1 - neg)
