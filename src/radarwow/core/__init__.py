"""radar-wow core: data model, band configuration, I/O, test data and diagnostics."""

from .algorithm import Algorithm, MissingMomentError, Reference, list_algorithms, register
from .bands import Band, BandConfig, band_from_frequency, load_band
from .data import SAMPLES, fmi_aws_files, load_sample, nexrad_aws, nexrad_aws_files
from .diagnose import diagnose
from .io import open_volume, sweep_names
from .moments import CANONICAL, standardize

__version__ = "0.1.0"
__all__ = ["Algorithm", "Band", "BandConfig", "CANONICAL", "MissingMomentError", "Reference", "SAMPLES",
           "band_from_frequency", "diagnose", "fmi_aws_files", "list_algorithms", "load_band", "load_sample", "nexrad_aws", "nexrad_aws_files",
           "open_volume", "register", "standardize", "sweep_names"]
