"""Health check for any radar volume: find the data problems *before* running algorithms.

Different radars have different problems (missing moments, PHIDP wrapping and system
offset, ZDR bias, rhoHV miscalibration, pre-applied filtering, interference rays,
irregular azimuths...). `diagnose` measures each one on the lowest dual-pol sweep and
points to the radar-wow stage/algorithm that deals with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .io import sweep_names
from .moments import PROCESSED_HINTS, phidp_direction
from .noise import noise_floor, signal_mask

SEVERITY_ORDER = {"blocker": 0, "fix": 1, "check": 2, "info": 3}


@dataclass
class Finding:
    severity: str          # blocker | fix | check | info
    topic: str
    message: str
    remedy: str = ""


@dataclass
class Report:
    source: str
    band: str
    sweep: str
    findings: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)

    def add(self, *a, **k):
        self.findings.append(Finding(*a, **k))

    def to_markdown(self) -> str:
        rows = ["| severity | topic | finding | remedy |", "|---|---|---|---|"]
        for f in sorted(self.findings, key=lambda f: SEVERITY_ORDER[f.severity]):
            rows.append(f"| {f.severity} | {f.topic} | {f.message} | {f.remedy} |")
        return f"**{self.source}** (band {self.band}, {self.sweep})\n\n" + "\n".join(rows)

    def _repr_markdown_(self):
        return self.to_markdown()


def _lowest_sweep(dt: xr.DataTree, prefer=("PHIDP", "RHOHV", "DBZH")) -> str:
    names = sweep_names(dt)
    scored = [(sum(m in dt[n].ds for m in prefer), -i, n) for i, n in enumerate(names)]
    return max(scored)[2]


def diagnose(dt: xr.DataTree, sweep: str | None = None) -> Report:
    sweep = sweep or _lowest_sweep(dt)
    ds = dt[sweep].to_dataset()
    rep = Report(source=dt.attrs.get("radarwow_sample", dt.attrs.get("instrument_name", "volume")),
                 band=str(dt.attrs.get("radarwow_band")), sweep=sweep)
    v = set(ds.data_vars)

    # --- moments ----------------------------------------------------------------
    core = ["DBZH", "ZDR", "PHIDP", "RHOHV", "VRADH"]
    rep.numbers["moments"] = [m for m in core if m in v]
    for m, why in [("PHIDP", "no KDP, no PHIDP-based attenuation correction, no R(KDP)/R(A)"),
                   ("RHOHV", "no dual-pol QC or hydrometeor classification"),
                   ("ZDR", "no R(Z,ZDR), weaker classification")]:
        if m not in v:
            rep.add("blocker", "moments", f"{m} missing: {why}", "request the moment from the operator; fall back to single-pol chain")
    processed = sorted(x for x in v if x.startswith(PROCESSED_HINTS))
    if processed:
        rep.add("info", "provenance", f"{len(processed)} pre-processed fields in file (e.g. {', '.join(processed[:3])})",
                "test algorithms on raw moments; use these only as a reference")
    sent = {m: ds[m].attrs["radarwow_codes_removed"] for m in v if "radarwow_codes_removed" in ds[m].attrs}
    if sent:
        rep.add("info", "censoring", "NEXRAD below-threshold / range-folded codes set to missing: "
                + ", ".join(f"{m}={c[0]:g},{c[1]:g}" for m, c in sent.items()), "handled by core.open_volume")
    if "SNRH" in v and "SQIH" in v:
        a, b = float(np.nanmean(ds["SNRH"].values > 3)), float(np.nanmean(ds["SQIH"].values > 0.4))
        if abs(a - b) > 0.3:
            rep.add("check", "noise", f"SNR > 3 dB on {a:.0%} of gates but NCP/SQI > 0.4 on only {b:.0%}: SNR field looks biased",
                    "signal mask uses both (core.noise.signal_mask)")
    if "DBTH" in v and "DBZH" in v:
        rep.add("info", "clutter", "both total (DBTH) and filtered (DBZH) reflectivity present",
                "DBTH - DBZH = clutter power removed by the signal processor")

    # --- geometry -----------------------------------------------------------------
    az = np.sort(np.mod(ds["azimuth"].values, 360))
    d = np.diff(np.concatenate([az, [az[0] + 360]]))
    res = float(np.median(d))
    gaps = int(np.sum(d > 1.5 * res))
    dups = int(np.sum(d < 0.25 * res))
    rng = ds["range"].values
    dr = float(np.median(np.diff(rng)))
    rep.numbers.update(n_rays=int(az.size), az_res=round(res, 3), gate_m=round(dr, 1), max_range_km=round(float(rng[-1]) / 1e3, 1))
    if gaps or dups:
        rep.add("fix", "geometry", f"{gaps} azimuth gaps and {dups} duplicate rays (resolution {res:.2f} deg)",
                "regrid azimuths before texture/KDP filters (core.geometry)")

    def arr(m):
        return ds[m].values.astype("float64") if m in v else None

    z, zdr, phi, rho = arr("DBZH"), arr("ZDR"), arr("PHIDP"), arr("RHOHV")

    # --- noise and reflectivity ------------------------------------------------------
    sig = signal_mask(ds) if z is not None else None
    if z is not None:
        _, n0, censored = noise_floor(z, rng)
        rep.numbers["noise_censored"] = censored
        rep.numbers["frac_signal"] = round(float(sig.mean()), 3)
        if not censored:
            rep.add("fix", "noise", f"noise gates are stored, not censored (floor ~{n0:.0f} dBZ at 1 km, rising 20 log r); "
                    f"{sig.mean():.0%} of gates are signal", "qc: signal mask (SNR or noise-floor + 3 dB) before any other step")
        zs = np.where(sig, z, np.nan)
        if sig.any():
            rep.numbers["frac_Z_gt_60"] = round(float(np.nanmean(zs > 60)), 4)
            if np.nanmean(zs > 60) > 0.002:
                rep.add("check", "clutter", f"{np.nanmean(zs > 60):.2%} of signal gates > 60 dBZ (clutter or hail)",
                        "qc: clutter detection (Hubbert 2009 textures)")
        # interference-like rays: signal on nearly every gate beyond 30 km
        far = rng > 30e3
        if far.sum() > 20 and sig.any():
            frac = sig[:, far].mean(axis=1)
            ref = np.median(frac)
            spike = (frac > 0.9) & (frac > ref + 0.4)
            if "RHOHV" in v:
                with np.errstate(all="ignore"):
                    import warnings
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rmed = np.nanmedian(np.where(sig[:, far], ds["RHOHV"].values[:, far], np.nan), axis=1)
                spike &= rmed < 0.85  # emitter energy is noise-like: low rhoHV
            rep.numbers["interference_rays"] = int(spike.sum())
            if spike.any():
                rep.add("fix", "interference", f"{int(spike.sum())} rays with signal on > 90 % of gates beyond 30 km (emitter-spike signature)",
                        "qc: interference detection (Saltikoff 2016 signature)")

    # --- rhoHV -------------------------------------------------------------------
    rain = None
    if rho is not None:
        rs = rho[sig] if sig is not None else rho.ravel()
        rep.numbers["rhohv_gt_1.02"] = round(float(np.nanmean(rs > 1.02)), 4)
        if np.nanmean(rs > 1.02) > 0.01:
            rep.add("check", "rhoHV", f"{np.nanmean(rs > 1.02):.1%} of signal gates rhoHV > 1.02 (noise-correction/calibration issue)",
                    "check SNR-dependent rhoHV correction")
        rain = (rho > 0.97) & (z > 15) & sig if z is not None else (rho > 0.97)
        rep.numbers["rhohv_median_rain"] = round(float(np.nanmedian(rho[rain])), 3) if rain.any() else None
        if rain.any() and np.nanmedian(rho[rain]) < 0.97:
            rep.add("check", "rhoHV", "median rhoHV in rain < 0.97: low SNR or miscalibration", "check noise correction")

    # --- PHIDP ---------------------------------------------------------------------
    if phi is not None:
        ps = phi[rain] if rain is not None and rain.any() else phi
        lo, hi = float(np.nanpercentile(ps, 0.5)), float(np.nanpercentile(ps, 99.5))
        rep.numbers["phidp_range"] = (round(lo, 1), round(hi, 1))
        if lo < -90:
            rep.add("fix", "PHIDP", f"PHIDP in rain spans {lo:.0f} to {hi:.0f} deg (signed, wraps at +-180)",
                    "kdp: unfold before differentiation (Vulpiani 2012 folding test)")
        good = np.isfinite(phi) & (rain if rain is not None else np.isfinite(phi))
        first = []
        jumps = 0
        for i in range(phi.shape[0]):
            idx = np.flatnonzero(good[i])
            if idx.size >= 10:
                first.append(np.nanmedian(phi[i, idx[:10]]))
                seg = phi[i, idx]
                jumps += int(np.any(np.diff(seg) < -180))
        if first:
            off = float(np.nanmedian(first))
            rep.numbers["phidp_system_offset"] = round(off, 1)
            rep.numbers["rays_with_folding"] = jumps
            rep.add("fix" if abs(off) > 10 else "info", "PHIDP", f"system PHIDP offset ~{off:.0f} deg (median of first rain gates)",
                    "kdp: subtract offset / use offset-free derivative (Vulpiani 2012)")
            if jumps:
                rep.add("fix", "PHIDP", f"{jumps} rays with a > 180 deg drop (folding)", "kdp: unfold (Vulpiani 2012)")

    zdr_clipped = False
    if zdr is not None and sig is not None and np.isfinite(zdr[sig]).sum() > 1000:
        if np.nanmin(zdr[sig]) >= 0:
            zdr_clipped = True
            rep.add("fix", "ZDR", "ZDR is never negative on signal gates: clipped or stored unsigned",
                    "treat ZDR as unusable for bias checks and R(Z,ZDR) until the storage is understood")
    if phi is not None:
        sign, agree = phidp_direction(ds)
        rep.numbers["phidp_direction"] = sign
        if sign < 0:
            rep.add("fix", "PHIDP", f"PHIDP decreases with range in rain on {agree:.0%} of rays (reversed sign convention)",
                    "kdp: flip sign before processing (done automatically in radarwow.kdp)")
    # --- ZDR light-rain check (Ryzhkov et al. 2005 idea) ---------------------------
    if zdr is not None and z is not None and rho is not None and not zdr_clipped:
        sel = (z >= 20) & (z <= 28) & (rho > 0.98) & np.isfinite(zdr) & sig
        if "height_over_iso0" in ds:
            sel &= ds["height_over_iso0"].values < -500
        if sel.sum() > 200:
            med = float(np.nanmedian(zdr[sel]))
            rep.numbers["zdr_median_light_rain"] = round(med, 2)
            if abs(med - 0.25) > 0.3:
                rep.add("fix", "ZDR", f"median ZDR at 20-28 dBZ, rhoHV > 0.98 = {med:+.2f} dB; expected ~ +0.2-0.3 dB *if this is rain* (snow plates/dendrites can be higher)",
                        "calibration: ZDR offset (Ryzhkov et al. 2005); confirm with vertical scans or Sun")
            else:
                rep.add("info", "ZDR", f"median ZDR in light rain {med:+.2f} dB: no large bias", "")
        else:
            rep.add("info", "ZDR", "too few light-rain gates for the ZDR bias check on this sweep", "")
    return rep
