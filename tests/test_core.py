import numpy as np
import pytest
import xarray as xr

from radarwow.core import Algorithm, Band, MissingMomentError, Reference, band_from_frequency, load_band, standardize
from radarwow.core.diagnose import diagnose
from radarwow.core.noise import mask_nexrad_codes, signal_mask
from radarwow.core.testing import synthetic_ray, synthetic_sweep


def test_band_from_frequency():
    assert band_from_frequency(2.8e9) is Band.S
    assert band_from_frequency(5.6e9) is Band.C
    assert band_from_frequency(9.4e9) is Band.X


@pytest.mark.parametrize("band", ["S", "C", "X"])
def test_band_configs_have_sources(band):
    cfg = load_band(band)
    assert cfg.params["kdp_min"] < 0 < cfg.params["kdp_max"]
    assert all(cfg.sources[k] for k in cfg.params)


def test_standardize_prefers_raw_over_corrected():
    ds = xr.Dataset({"reflectivity": ("x", [1.0]), "corrected_reflectivity": ("x", [2.0]),
                     "differential_phase": ("x", [3.0]), "copol_correlation_coeff": ("x", [0.9])})
    out, mapping = standardize(ds)
    assert mapping["DBZH"] == "reflectivity" and float(out["DBZH"][0]) == 1.0
    assert "corrected_reflectivity" in out and mapping["PHIDP"] == "differential_phase" and "RHOHV" in out


def test_nexrad_codes_masked():
    da = xr.DataArray(np.array([-33.0, -32.5, -32.0, 10.0, 20.0, -33.0]))
    out, codes = mask_nexrad_codes(da)
    assert codes == [-33.0, -32.5]
    assert np.isnan(out.values[[0, 1, 5]]).all() and out.values[2] == -32.0


def test_algorithm_checks_moments_and_band():
    class Dummy(Algorithm):
        name = "dummy"
        requires = ("PHIDP",)
        bands = frozenset({Band.C})
        reference = Reference("A. Author", 2020, "t", "j")

        def run_sweep(self, ds, cfg):
            return ds.assign(OUT=ds["PHIDP"] * 0)

    ds = synthetic_sweep()
    assert "OUT" in Dummy()(ds, band="C")
    with pytest.raises(ValueError):
        Dummy()(ds, band="S")
    with pytest.raises(MissingMomentError):
        Dummy()(ds.drop_vars("PHIDP"), band="C")


def test_signal_mask_and_spike_diagnosis():
    ds = synthetic_sweep(spike_az=300)
    sig = signal_mask(ds)
    assert sig[300].mean() > 0.9
    dt = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"radarwow_band": "C"}), "sweep_0": ds})
    rep = diagnose(dt)
    assert rep.numbers["interference_rays"] >= 1
    assert any(f.topic == "PHIDP" and "offset" in f.message for f in rep.findings)


def test_synthetic_ray_truth():
    r, psi, phi, kdp = synthetic_ray(noise=0.0)
    assert np.allclose(psi, phi)
    assert np.isclose(phi[-1] - phi[0], 2 * kdp.sum() * 0.25, rtol=1e-6)


def test_signal_mask_snr_trust_is_opt_in():
    import numpy as np
    import xarray as xr

    from radarwow.core.noise import signal_mask

    ds = xr.Dataset({"DBZH": (("azimuth", "range"), np.full((1, 3), 30.0)), "SNRH": (("azimuth", "range"), [[40.0, 40.0, 5.0]]),
                     "SQIH": (("azimuth", "range"), [[0.9, 0.2, 0.2]])})
    assert signal_mask(ds).tolist() == [[True, False, False]]
    ds.attrs["radarwow_snr_trust_db"] = 15.0
    assert signal_mask(ds).tolist() == [[True, True, False]]
