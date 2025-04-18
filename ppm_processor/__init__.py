"""
Polychromatic Polarization Microscope (PPM) Package

A package for processing polychromatic polarization microscopy images.
"""

from importlib.metadata import version, PackageNotFoundError
from .ppm_process import (
    PolychromaticPolarizationProcessor,
    ProcessingParameters,
    ImageData,
    PolarizationPair,
    main
)
from .image_corrections import (
    flatfield_correction,
    white_balance,
    non_uniform_illumination_correction,
    load_correction_images,
    apply_corrections
)

try:
    __version__ = version("ppm")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.2.0"  # fallback version 