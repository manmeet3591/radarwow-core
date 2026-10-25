"""Common interface for every radar-wow algorithm.

An algorithm works on one sweep (an ``xarray.Dataset`` with ``azimuth`` x ``range``
dimensions and canonical moment names) and returns a dataset with new or modified
variables. ``Algorithm.__call__`` also accepts a whole volume (``xarray.DataTree``)
and applies itself to every ``sweep_*`` node.

Each algorithm declares the paper it implements, the bands it supports, the moments
it needs and the ones it produces, so the pipeline can check compatibility before
running and record provenance afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import xarray as xr

from .bands import Band, BandConfig, load_band
from .moments import missing


@dataclass(frozen=True)
class Reference:
    authors: str
    year: int
    title: str
    journal: str
    doi: str = ""

    def short(self) -> str:
        first = self.authors.split(",")[0].split(" and ")[0].strip()
        etal = " et al." if ("," in self.authors or " and " in self.authors) else ""
        return f"{first}{etal} {self.year}"


class MissingMomentError(KeyError):
    pass


class Algorithm:
    """Base class. Subclasses set the class attributes and implement `run_sweep`.

    Constructor keyword arguments override entries of the class-level ``params`` dict.
    """

    name: str = ""
    stage: str = ""
    reference: Reference | None = None
    bands: frozenset = frozenset({Band.S, Band.C, Band.X})
    requires: tuple = ()
    optional: tuple = ()
    provides: tuple = ()
    notes: str = ""
    params: dict = {}

    def __init__(self, **params):
        self.params = {**type(self).params, **params}

    def __repr__(self):
        return f"{type(self).__name__}({self.params})"

    # -- to implement -----------------------------------------------------------
    def run_sweep(self, ds: xr.Dataset, cfg: BandConfig) -> xr.Dataset:  # pragma: no cover
        raise NotImplementedError

    # -- driver -----------------------------------------------------------------
    def __call__(self, data, band: Band | str | BandConfig | None = None, **params):
        cfg = self._config(data, band)
        if params:
            self.params = {**self.params, **params}
        cfg = cfg.override(**{k: v for k, v in self.params.items() if k in cfg.params})
        if isinstance(data, xr.DataTree):
            out = data.copy()
            for name, node in data.children.items():
                if name.startswith("sweep"):
                    out[name] = xr.DataTree(self._run_checked(node.to_dataset(), cfg))
            return out
        return self._run_checked(data, cfg)

    def _config(self, data, band) -> BandConfig:
        if isinstance(band, BandConfig):
            cfg = band
        else:
            if band is None:
                root = data if isinstance(data, xr.Dataset) else data.to_dataset()
                band = root.attrs.get("radarwow_band")
                if band is None:
                    raise ValueError(f"{self.name}: pass band='S'|'C'|'X' (not found in attrs)")
            cfg = load_band(band)
        if cfg.band not in self.bands:
            raise ValueError(f"{self.name} supports bands {sorted(b.value for b in self.bands)}, got {cfg.band.value}")
        return cfg

    def _run_checked(self, ds: xr.Dataset, cfg: BandConfig) -> xr.Dataset:
        miss = missing(ds, self.requires)
        if miss:
            raise MissingMomentError(f"{self.name} needs {miss}; sweep has {sorted(ds.data_vars)}")
        out = self.run_sweep(ds, cfg)
        stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ} {self.name} ({self.reference.short() if self.reference else ''}) band={cfg.band.value}"
        out.attrs["radarwow_history"] = (ds.attrs.get("radarwow_history", "") + "\n" + stamp).strip()
        return out

    def describe(self) -> str:
        ref = self.reference
        lines = [f"{self.name} [{self.stage}]"]
        if ref:
            lines.append(f"  {ref.authors} ({ref.year}). {ref.title}. {ref.journal}. https://doi.org/{ref.doi}")
        lines.append(f"  bands: {', '.join(sorted(b.value for b in self.bands))}")
        lines.append(f"  requires: {', '.join(self.requires) or '-'}   optional: {', '.join(self.optional) or '-'}")
        lines.append(f"  provides: {', '.join(self.provides)}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)


REGISTRY: dict[str, type[Algorithm]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def list_algorithms(stage: str | None = None, band: Band | str | None = None):
    out = []
    for name, cls in REGISTRY.items():
        a = cls()
        if stage and a.stage != stage:
            continue
        if band and Band(band) not in a.bands:
            continue
        out.append(a)
    return out
