"""Frequency bands and band-dependent parameters.

Parameters live in ``bands/<BAND>.toml`` with a source for every number, so an
algorithm never hard-codes S/C/X constants. Override any value per radar with
``BandConfig.override(**kw)``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import resources


class Band(str, Enum):
    S = "S"
    C = "C"
    X = "X"


#: nominal frequency ranges in GHz (IEEE radar bands)
RANGES_GHZ = {Band.S: (2.0, 4.0), Band.C: (4.0, 8.0), Band.X: (8.0, 12.0)}

C_LIGHT = 299_792_458.0


def band_from_frequency(freq_hz: float) -> Band:
    ghz = freq_hz / 1e9
    for b, (lo, hi) in RANGES_GHZ.items():
        if lo <= ghz < hi:
            return b
    raise ValueError(f"{ghz:.2f} GHz is outside S/C/X")


def band_from_wavelength(wavelength_m: float) -> Band:
    return band_from_frequency(C_LIGHT / wavelength_m)


@dataclass(frozen=True)
class BandConfig:
    band: Band
    frequency_ghz: float
    params: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.params[key]

    def get(self, key, default=None):
        return self.params.get(key, default)

    def override(self, **kw) -> "BandConfig":
        return replace(self, params={**self.params, **kw})


def load_band(band: Band | str) -> BandConfig:
    band = Band(band)
    text = resources.files("radarwow.core").joinpath(f"bands/{band.value}.toml").read_text()
    d = tomllib.loads(text)
    params = {k: v["value"] for k, v in d["params"].items()}
    sources = {k: v.get("source", "") for k, v in d["params"].items()}
    return BandConfig(band=band, frequency_ghz=d["frequency_ghz"], params=params, sources=sources)
