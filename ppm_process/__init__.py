"""
Polychromatic Polarization Microscope (PPM) Package

A package for processing polychromatic polarization microscopy images.
"""

from importlib.metadata import version, PackageNotFoundError
from .ppm_process import main

try:
    __version__ = version("ppm")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.2.0"  # fallback version 