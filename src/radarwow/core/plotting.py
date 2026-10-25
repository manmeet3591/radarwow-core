"""Quick-look plots shared by all notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

STYLE = {
    "DBZH": dict(cmap="turbo", vmin=-10, vmax=65, label="Z (dBZ)"),
    "DBTH": dict(cmap="turbo", vmin=-10, vmax=65, label="Z total (dBZ)"),
    "ZDR": dict(cmap="RdYlBu_r", vmin=-2, vmax=5, label="ZDR (dB)"),
    "PHIDP": dict(cmap="twilight", vmin=-180, vmax=180, label="PHIDP (deg)"),
    "RHOHV": dict(cmap="viridis", vmin=0.7, vmax=1.0, label="rhoHV"),
    "KDP": dict(cmap="magma_r", vmin=-1, vmax=5, label="KDP (deg/km)"),
    "VRADH": dict(cmap="RdBu_r", vmin=-30, vmax=30, label="V (m/s)"),
}


def xy_km(ds: xr.Dataset):
    az = np.deg2rad(ds["azimuth"].values)[:, None]
    r = ds["range"].values[None, :] / 1e3
    el = np.deg2rad(float(np.nanmean(ds["elevation"].values))) if "elevation" in ds else 0.0
    return r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az)


def ppi(ds: xr.Dataset, var: str, ax=None, max_km: float | None = None, title: str | None = None, **kw):
    ax = ax or plt.gca()
    st = {**STYLE.get(var, dict(cmap="viridis", label=var)), **kw}
    label = st.pop("label")
    x, y = xy_km(ds)
    data = ds[var].values if var in ds else np.full(x.shape, np.nan)
    m = ax.pcolormesh(x, y, np.ma.masked_invalid(data), shading="auto", **st)
    ax.set_aspect("equal")
    if max_km:
        ax.set_xlim(-max_km, max_km)
        ax.set_ylim(-max_km, max_km)
    ax.set_title(title or var, fontsize=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    plt.colorbar(m, ax=ax, label=label, shrink=0.8)
    return m


def panel(ds: xr.Dataset, variables, max_km=None, suptitle=None, ncols=None, size=4.2):
    ncols = ncols or len(variables)
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows * 0.85), squeeze=False)
    for ax, var in zip(axes.flat, variables):
        if isinstance(var, tuple):
            var, title = var
        else:
            title = var
        ppi(ds, var, ax=ax, max_km=max_km, title=title)
    for ax in list(axes.flat)[len(variables):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def ray(ds: xr.Dataset, azimuth: float, variables, ax=None):
    i = int(np.argmin(np.abs(((ds["azimuth"].values - azimuth + 180) % 360) - 180)))
    ax = ax or plt.gca()
    r = ds["range"].values / 1e3
    for v in variables:
        ax.plot(r, ds[v].values[i], label=v)
    ax.set_xlabel("range (km)")
    ax.legend()
    ax.set_title(f"ray at azimuth {float(ds['azimuth'].values[i]):.1f} deg", fontsize=10)
    return i
