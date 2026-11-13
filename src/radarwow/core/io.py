"""Open any supported radar file as an ``xarray.DataTree`` with canonical moment names.

Uses xradar for standard formats (NEXRAD Level II, ODIM HDF5, CfRadial 1/2, IRIS,
Rainbow, GAMIC, UF, ...) and adds an adapter for IMD's IRIS-exported FSL netCDF
(one sweep per file), which xradar does not read.
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import xradar as xd

from .bands import Band, band_from_frequency, band_from_wavelength
from .moments import standardize
from .noise import mask_nexrad_codes

OPENERS = {
    "nexrad": xd.io.open_nexradlevel2_datatree,
    "odim": xd.io.open_odim_datatree,
    "cfradial1": xd.io.open_cfradial1_datatree,
    "cfradial2": xd.io.open_cfradial2_datatree,
    "iris": xd.io.open_iris_datatree,
    "rainbow": xd.io.open_rainbow_datatree,
    "gamic": xd.io.open_gamic_datatree,
    "uf": xd.io.open_uf_datatree,
}


def _gunzip(path: str) -> str:
    out = path[:-3]
    if not os.path.exists(out):
        with gzip.open(path) as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return out


def detect_format(path: str) -> str:
    p = path.lower()
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:4] == b"AR2V" or head[:8] == b"ARCHIVE2":
        return "nexrad"
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return "odim" if not p.endswith((".nc", ".cdf")) else "cfradial1"
    if head[:3] == b"CDF" or p.endswith((".nc", ".cdf")):
        with xr.open_dataset(path) as ds:
            if "radialAzim" in ds and "gateSize" in ds:
                return "imd_fsl"
            if "sweep_start_ray_index" in ds.variables:
                return "cfradial1"
        return "cfradial2"
    if p.endswith(".vol"):
        return "rainbow"
    if p.endswith(".uf"):
        return "uf"
    return "iris"


def open_imd_fsl(paths) -> xr.DataTree:
    """IMD IRIS->netCDF (FSL convention): one file per sweep with T, Z, V, W, ZDR, HCLASS."""
    paths = sorted(paths) if not isinstance(paths, (str, os.PathLike)) else [paths]
    sweeps, meta = [], None
    for path in paths:
        with xr.open_dataset(path) as raw:
            raw = raw.load()
        rng = float(raw["firstGateRange"]) + np.arange(raw.sizes["bin"]) * float(raw["gateSize"])
        moments = [v for v in ("T", "Z", "V", "W", "ZDR", "HCLASS") if v in raw]
        ds = xr.Dataset(
            {m: (("azimuth", "range"), raw[m].values.astype("float32")) for m in moments},
            coords={"azimuth": ("azimuth", raw["radialAzim"].values.astype("float64")),
                    "range": ("range", rng),
                    "elevation": ("azimuth", raw["radialElev"].values.astype("float64")),
                    "time": ("azimuth", raw["radialTime"].values)},
        ).sortby("azimuth")
        ds["sweep_fixed_angle"] = float(raw["elevationAngle"])
        ds["sweep_mode"] = "azimuth_surveillance"
        sweeps.append((float(raw["elevationAngle"]), ds))
        meta = raw
    sweeps.sort(key=lambda t: t[0])
    tree = {"/": xr.Dataset(attrs={"source": "IMD IRIS netCDF (FSL)", "instrument_name": "IMD DWR"})}
    tree["/"]["latitude"] = float(meta["siteLat"])
    tree["/"]["longitude"] = float(meta["siteLon"])
    tree["/"]["altitude"] = float(meta["siteAlt"])
    for i, (_, ds) in enumerate(sweeps):
        ds = ds.assign_coords(latitude=float(meta["siteLat"]), longitude=float(meta["siteLon"]),
                              altitude=float(meta["siteAlt"]))
        tree[f"sweep_{i}"] = ds
    dt = xr.DataTree.from_dict(tree)
    dt.attrs["wavelength_m"] = float(meta["waveLength"]) / 100.0  # file stores cm
    return dt


def open_volume(path, fmt: str = "auto", band: Band | str | None = None, sweeps=None) -> xr.DataTree:
    """Open a volume, standardize moment names on every sweep, and record the band.

    ``path`` may be a single file or (IMD) a list of per-sweep files.
    """
    if not isinstance(path, (str, os.PathLike)):
        fmt = "imd_fsl"
    else:
        path = str(path)
        if path.endswith(".gz"):
            path = _gunzip(path)
        if fmt == "auto":
            fmt = detect_format(path)
    if fmt == "imd_fsl":
        dt = open_imd_fsl(path)
    else:
        kw = {"sweep": sweeps} if sweeps is not None else {}
        try:
            dt = OPENERS[fmt](path, **kw)
        except (IndexError, KeyError, ValueError) as err:
            if sweeps is not None:
                raise
            dt = _open_sweepwise(OPENERS[fmt], path, err)
    out = {"/": dt.to_dataset()}
    mappings = {}
    for name, node in dt.children.items():
        ds = node.to_dataset()
        if name.startswith("sweep"):
            ds, mappings[name] = standardize(ds)
            if fmt == "nexrad":
                for m in ("DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH", "WRADH"):
                    if m in ds:
                        ds[m], _ = mask_nexrad_codes(ds[m])
            if fmt == "odim":
                ds.attrs["radarwow_snr_trust_db"] = 15.0  # see noise.signal_mask (FMI SQI is low in rain)
        out[name] = ds
    tree = xr.DataTree.from_dict(out)
    tree.attrs.update(dt.attrs)
    tree.attrs["radarwow_format"] = fmt
    tree.attrs["radarwow_band"] = Band(band).value if band else _guess_band(tree, fmt)
    return tree


def _open_sweepwise(opener, path, err, max_sweeps: int = 40, stop_after: int = 3):
    """Open sweep by sweep when the whole file fails (truncated or partly corrupt archives).

    Damaged sweeps are skipped and listed in the root attribute ``radarwow_bad_sweeps``; reading stops
    after ``stop_after`` consecutive failures past the last good sweep (the end of a truncated file).
    """
    good, bad, misses = {}, [], 0
    root = None
    for i in range(max_sweeps):
        try:
            part = opener(path, sweep=[i])
        except Exception:  # noqa: BLE001 - any decoder error marks this sweep as damaged
            bad.append(i)
            misses += 1
            if good and misses >= stop_after:
                break
            continue
        misses = 0
        root = root if root is not None else part.to_dataset()
        for name, node in part.children.items():
            if name.startswith("sweep"):
                good[name] = node.to_dataset()
    if not good:
        raise err
    last = max(int(n.split("_")[1]) for n in good)
    tree = xr.DataTree.from_dict({"/": root, **good})
    tree.attrs["radarwow_bad_sweeps"] = [i for i in bad if i < last]
    tree.attrs["radarwow_last_sweep"] = last
    return tree


def _guess_band(dt: xr.DataTree, fmt: str) -> str | None:
    if fmt == "nexrad":
        return Band.S.value
    if "wavelength_m" in dt.attrs:
        return band_from_wavelength(dt.attrs["wavelength_m"]).value
    for node in dt.subtree:
        ds = node.to_dataset()
        for v in ("frequency", "radar_frequency"):
            if v in ds and np.size(ds[v]):
                return band_from_frequency(float(np.ravel(ds[v].values)[0])).value
    return None


def sweep_names(dt: xr.DataTree) -> list[str]:
    return sorted((k for k in dt.children if k.startswith("sweep")), key=lambda s: int(s.split("_")[1]))
