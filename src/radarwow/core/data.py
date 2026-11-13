"""Free test data for every band, with the known problems of each source.

All files come from the openradar ``open-radar-data`` collection (downloaded and cached
by pooch) or from the public NEXRAD Level II archive on AWS. ``known_issues`` lists the
data problems we found in each source; `radarwow.core.diagnose` detects them
automatically on any other file too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import xarray as xr

from .io import open_volume


@dataclass(frozen=True)
class Sample:
    name: str
    band: str
    files: tuple
    radar: str
    when: str
    description: str
    known_issues: tuple = field(default_factory=tuple)


SAMPLES = {s.name: s for s in [
    Sample("nexrad_ktlx_s", "S", ("KTLX20130520_201643_V06.gz",), "NWS WSR-88D KTLX (Oklahoma City)",
           "2013-05-20 20:16 UTC", "Moore tornado supercell; operational S-band dual-pol, 250 m gates, 0.5 deg azimuths.",
           ("Below-threshold and range-folded gates are stored as code values (DBZH -33/-32.5, ZDR -8, RHOHV 0.20): masked on read.",
            "PHIDP stored 0-360 deg with a ~36 deg system offset; 38 rays wrap past 360 (folding) in the lowest sweep.",
            "rhoHV > 1.02 on ~15 % of signal gates (no low-SNR cap).",
            "Reflectivity is already clutter-filtered (CMD) in the RDA; no total-power moment.",
            "Split cuts: reflectivity/dual-pol and velocity of the lowest tilt are separate sweeps.")),
    Sample("arm_csapr2_c", "C", ("corcsapr2cmacppiM1.c1.20181111.030003.nc",), "ARM CSAPR2 (Cordoba, Argentina, RELAMPAGO)",
           "2018-11-11 03:00 UTC", "ARM C-band scanning precipitation radar, 0.5 deg PPI, processed by ARM CMAC.",
           ("Raw moments *and* ARM-CMAC corrected copies (corrected_*, filtered_*) in one file: test on raw, use corrected only as reference.",
            "Noise gates stored (not censored); the SNR field is biased (SNR > 3 dB on 98 % of gates vs NCP > 0.4 on 45 %).",
            "PHIDP 0-360 deg with a ~197 deg system offset and folding on 36 rays; it *decreases* with range in rain on 92 % of rays (reversed sign convention).",
            "C-band attenuation and differential attenuation in convective cores; ground clutter near the Sierras de Cordoba.")),
    Sample("arm_xprecip_x", "X", ("gucxprecipradarcmacppiS2.c1.20220314.021559.nc",), "ARM X-band precipitation radar (Gunnison, Colorado, SAIL)",
           "2022-03-14 02:15 UTC", "ARM X-band dual-pol in complex terrain (Rocky Mountains), snow season.",
           ("Noise gates stored: only ~25 % of gates are signal; 22 duplicate rays (0.32 deg azimuths).",
            "PHIDP system offset ~102 deg; mostly snow/mixed phase so KDP is small and rain relations do not apply.",
            "Mountain beam blockage (cumulative blockage flag included); raw plus CMAC-corrected copies in one file.")),
    Sample("arm_xsapr_x", "X", ("swx_20120520_0641.nc",), "ARM X-SAPR (SGP, Oklahoma; Py-ART example file)",
           "2012-05-20 06:41 UTC", "X-band dual-pol volume (22 sweeps) through a convective line.",
           ("Older CfRadial with Py-ART-era field names; `diff_phase` is actually KDP (long_name specific_differential_phase_hv), PHIDP is `dp_phase_shift`.",
            "ZDR is never negative (clipped or unsigned storage): unusable for bias checks.",
            "Noise gates stored; PHIDP system offset ~119 deg with folding on 45 rays; heavy X-band attenuation behind cores.")),
    Sample("imd_jaipur_c", "C", tuple(["IMD/JPR220822135253-IMD-B.nc"] + [f"IMD/JPR220822135253-IMD-B.nc.{i}" for i in range(1, 10)]),
           "IMD Jaipur DWR (Vaisala C-band)", "2022-08-22 13:52 UTC",
           "Operational IMD C-band, IRIS exported to FSL netCDF, one sweep per file, 10 tilts.",
           ("No PHIDP, RHOHV or KDP exported: dual-pol QC, KDP and attenuation correction are impossible.",
            "Vendor HCLASS field present; Z already processed by IRIS (clutter filter, thresholds).",
            "Non-standard netCDF (FSL) needs the radar-wow adapter.")),
]}


def fetch(name: str):
    import open_radar_data as ord_

    s = SAMPLES[name]
    paths = [ord_.DATASETS.fetch(f) for f in s.files]
    return paths if len(paths) > 1 else paths[0]


def load_sample(name: str, sweeps=None) -> xr.DataTree:
    s = SAMPLES[name]
    dt = open_volume(fetch(name), band=s.band, sweeps=sweeps)
    dt.attrs["radarwow_sample"] = name
    return dt


def nexrad_aws(site: str, when, sweeps=None) -> xr.DataTree:
    """Nearest-in-time NEXRAD Level II volume from the public AWS bucket (no account needed)."""
    import os
    import tempfile
    from datetime import datetime

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    when = datetime.fromisoformat(str(when)) if not hasattr(when, "year") else when
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, connect_timeout=10, read_timeout=60,
                                          retries={"max_attempts": 3}))
    bucket = "unidata-nexrad-level2"
    prefix = f"{when:%Y/%m/%d}/{site}/"
    keys = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in p.get("Contents", []) if not o["Key"].endswith("_MDM")]
    if not keys:
        raise FileNotFoundError(f"no NEXRAD files for {site} on {when:%Y-%m-%d}")

    def t(k):
        stamp = os.path.basename(k)[4:19]
        return abs((datetime.strptime(stamp, "%Y%m%d_%H%M%S") - when.replace(tzinfo=None)).total_seconds())

    key = min(keys, key=t)
    local = os.path.join(tempfile.gettempdir(), "radarwow_nexrad", os.path.basename(key))
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        s3.download_file(bucket, key, local)
    dt = open_volume(local, band="S", sweeps=sweeps)
    dt.attrs["radarwow_sample"] = f"nexrad_aws:{os.path.basename(key)}"
    return dt


def nexrad_aws_files(site: str, start, end, cache_dir: str | None = None) -> list:
    """Download every NEXRAD Level II volume of ``site`` with start time in [start, end) (UTC).

    Files go to ``cache_dir`` (default ``~/.cache/radarwow/nexrad``) and are reused. Returns (time, path)
    pairs sorted by time. Hours missing from the archive (radar outages) are simply absent.
    """
    import os

    import boto3
    import pandas as pd
    from botocore import UNSIGNED
    from botocore.config import Config

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cache_dir = cache_dir or os.path.expanduser("~/.cache/radarwow/nexrad")
    os.makedirs(cache_dir, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, connect_timeout=10, read_timeout=120,
                                          retries={"max_attempts": 3}))
    out = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        prefix = f"{day:%Y/%m/%d}/{site}/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket="unidata-nexrad-level2", Prefix=prefix):
            for o in page.get("Contents", []):
                k = o["Key"]
                if k.endswith("_MDM"):
                    continue
                t = pd.to_datetime(os.path.basename(k)[4:19], format="%Y%m%d_%H%M%S")
                if start <= t < end:
                    local = os.path.join(cache_dir, os.path.basename(k))
                    if not os.path.exists(local):
                        s3.download_file("unidata-nexrad-level2", k, local)
                    out.append((t, local))
    return sorted(out)


def fmi_aws_files(site: str, start, end, cache_dir: str | None = None, every_min: int = 5) -> list:
    """Download Finnish Meteorological Institute ODIM HDF5 volumes (C band) of ``site`` (e.g. ``"fianj"``) with
    start time in [start, end) from the open AWS bucket ``fmi-opendata-radar-volume-hdf5`` (no account).

    Volumes are every 5 minutes, ~25 MB each; ``every_min`` (a multiple of 5) thins them. Returns sorted
    (time, path) pairs; files are cached in ``cache_dir`` (default ``~/.cache/radarwow/fmi``).
    """
    import os

    import boto3
    import pandas as pd
    from botocore import UNSIGNED
    from botocore.config import Config

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cache_dir = cache_dir or os.path.expanduser("~/.cache/radarwow/fmi")
    os.makedirs(cache_dir, exist_ok=True)
    s3 = boto3.client("s3", region_name="eu-west-1", config=Config(signature_version=UNSIGNED, connect_timeout=10, read_timeout=120,
                                                                   retries={"max_attempts": 3}))
    out = []
    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket="fmi-opendata-radar-volume-hdf5", Prefix=f"{day:%Y/%m/%d}/{site}/"):
            for o in page.get("Contents", []):
                k = o["Key"]
                name = os.path.basename(k)
                if not name.endswith("_PVOL.h5"):
                    continue
                t = pd.to_datetime(name[:12], format="%Y%m%d%H%M")
                if start <= t < end and t.minute % every_min == 0:
                    local = os.path.join(cache_dir, name)
                    if not os.path.exists(local):
                        s3.download_file("fmi-opendata-radar-volume-hdf5", k, local)
                    out.append((t, local))
    return sorted(out)


def ideam_aws_files(site: str, start, end, cache_dir: str | None = None, cycle_min: int = 5) -> list:
    """Download IDEAM (Colombia) radar files of ``site`` (e.g. ``"Tablazo"``) from the open AWS bucket
    ``s3-radaresideam`` (no account) and group them into volumes.

    The IRIS radars there write one sweep per file (plus a reflectivity-only volume task), every ``cycle_min``
    minutes; files whose time falls in the same cycle form one volume. Returns sorted (cycle start, [paths]) pairs,
    ready for `open_volume`. IMD radars use the same IRIS one-sweep-per-file layout.
    """
    import os

    import boto3
    import pandas as pd
    from botocore import UNSIGNED
    from botocore.config import Config

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cache_dir = cache_dir or os.path.expanduser("~/.cache/radarwow/ideam")
    os.makedirs(cache_dir, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED, connect_timeout=10, read_timeout=120,
                                                                   retries={"max_attempts": 3}))
    cycles: dict = {}
    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket="s3-radaresideam", Prefix=f"l2_data/{day:%Y/%m/%d}/{site}/"):
            for o in page.get("Contents", []):
                name = os.path.basename(o["Key"])
                if ".RAW" not in name or name.endswith(".gz"):
                    continue
                t = pd.to_datetime(name[3:15], format="%y%m%d%H%M%S")
                c = t.floor(f"{cycle_min}min")
                if start <= c < end:
                    local = os.path.join(cache_dir, name)
                    if not os.path.exists(local):
                        s3.download_file("s3-radaresideam", o["Key"], local)
                    cycles.setdefault(c, []).append(local)
    return sorted((c, sorted(p)) for c, p in cycles.items())
