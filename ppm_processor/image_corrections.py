"""
Image correction utilities for the PPM processor.

This module provides functions for various image corrections including:
- Flatfield correction
- White balancing
- Dark current subtraction
- Non-uniform illumination correction
"""

import numpy as np
from skimage import exposure
from typing import Tuple, Optional, Union, List
from pathlib import Path
import imageio


def flatfield_correction(
    image: np.ndarray,
    flatfield: np.ndarray,
    dark_current: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply flatfield correction to an image.
    
    Flatfield correction removes non-uniform illumination and sensor response variations.
    
    Args:
        image: Input image array (height, width, channels)
        flatfield: Flatfield image array (height, width, channels)
        dark_current: Optional dark current image array (height, width, channels)
        
    Returns:
        Corrected image array
    """
    if image.shape != flatfield.shape:
        raise ValueError("Image and flatfield must have the same shape")
    
    # Convert to float for calculations
    image_float = image.astype(np.float32)
    flatfield_float = flatfield.astype(np.float32)
    
    # Subtract dark current if provided
    if dark_current is not None:
        if image.shape != dark_current.shape:
            raise ValueError("Image and dark current must have the same shape")
        dark_current_float = dark_current.astype(np.float32)
        image_float = image_float - dark_current_float
        flatfield_float = flatfield_float - dark_current_float
    
    # Avoid division by zero
    flatfield_float = np.maximum(flatfield_float, 1e-10)
    
    # Apply flatfield correction
    corrected = image_float / flatfield_float
    
    # Normalize to [0, 1] range
    corrected = np.clip(corrected, 0, 1)
    
    return corrected


def white_balance(
    image: np.ndarray,
    reference_white: Optional[np.ndarray] = None,
    method: str = "gray_world"
) -> np.ndarray:
    """
    Apply white balancing to an image.
    
    Args:
        image: Input image array (height, width, channels)
        reference_white: Optional reference white values for each channel
        method: White balancing method, one of:
            - "gray_world": Assumes average color is gray
            - "white_patch": Uses brightest pixels as white reference
            - "manual": Uses provided reference_white values
            
    Returns:
        White-balanced image array
    """
    if method not in ["gray_world", "white_patch", "manual"]:
        raise ValueError("Method must be one of: gray_world, white_patch, manual")
    
    # Convert to float for calculations
    image_float = image.astype(np.float32)
    
    if method == "gray_world":
        # Gray world assumption: average color should be gray
        channel_means = np.mean(image_float, axis=(0, 1))
        mean_of_means = np.mean(channel_means)
        scale_factors = mean_of_means / np.maximum(channel_means, 1e-10)
        
    elif method == "white_patch":
        # Use brightest pixels as white reference
        # Find the top 1% brightest pixels
        threshold = np.percentile(np.max(image_float, axis=2), 99)
        bright_pixels = np.max(image_float, axis=2) > threshold
        
        # Calculate mean of bright pixels for each channel
        channel_means = np.zeros(image_float.shape[2])
        for i in range(image_float.shape[2]):
            channel_means[i] = np.mean(image_float[bright_pixels, i])
        
        # Scale to make the maximum channel equal to 1
        scale_factors = 1.0 / np.maximum(channel_means, 1e-10)
        
    else:  # manual
        if reference_white is None:
            raise ValueError("reference_white must be provided for manual white balancing")
        if len(reference_white) != image_float.shape[2]:
            raise ValueError("reference_white must have the same number of elements as image channels")
        
        scale_factors = 1.0 / np.maximum(reference_white, 1e-10)
    
    # Apply scaling factors
    balanced = image_float * scale_factors.reshape(1, 1, -1)
    
    # Normalize to [0, 1] range
    balanced = np.clip(balanced, 0, 1)
    
    return balanced


def non_uniform_illumination_correction(
    image: np.ndarray,
    illumination_map: Optional[np.ndarray] = None,
    method: str = "polynomial"
) -> np.ndarray:
    """
    Correct for non-uniform illumination in an image.
    
    Args:
        image: Input image array (height, width, channels)
        illumination_map: Optional pre-calculated illumination map
        method: Correction method, one of:
            - "polynomial": Fit a polynomial surface to estimate illumination
            - "gaussian": Use Gaussian blur to estimate illumination
            - "manual": Use provided illumination_map
            
    Returns:
        Corrected image array
    """
    if method not in ["polynomial", "gaussian", "manual"]:
        raise ValueError("Method must be one of: polynomial, gaussian, manual")
    
    # Convert to float for calculations
    image_float = image.astype(np.float32)
    
    if method == "manual":
        if illumination_map is None:
            raise ValueError("illumination_map must be provided for manual correction")
        if illumination_map.shape != image_float.shape:
            raise ValueError("illumination_map must have the same shape as the image")
        
        # Apply illumination correction
        corrected = image_float / np.maximum(illumination_map, 1e-10)
        
    else:
        # Process each channel separately
        corrected = np.zeros_like(image_float)
        
        for i in range(image_float.shape[2]):
            channel = image_float[:, :, i]
            
            if method == "polynomial":
                # Fit a polynomial surface to estimate illumination
                from scipy.optimize import curve_fit
                
                # Create coordinate arrays
                y, x = np.mgrid[0:channel.shape[0], 0:channel.shape[1]]
                xy = np.vstack((x.ravel(), y.ravel())).T
                
                # Define polynomial function (2nd order)
                def poly_func(xy, a, b, c, d, e, f):
                    x, y = xy.T
                    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y
                
                # Fit polynomial to the image
                try:
                    popt, _ = curve_fit(poly_func, xy, channel.ravel())
                    illumination = poly_func(xy, *popt).reshape(channel.shape)
                except:
                    # Fallback to Gaussian if polynomial fitting fails
                    from scipy.ndimage import gaussian_filter
                    illumination = gaussian_filter(channel, sigma=50)
                
            else:  # gaussian
                # Use Gaussian blur to estimate illumination
                from scipy.ndimage import gaussian_filter
                illumination = gaussian_filter(channel, sigma=50)
            
            # Apply illumination correction
            corrected[:, :, i] = channel / np.maximum(illumination, 1e-10)
    
    # Normalize to [0, 1] range
    corrected = np.clip(corrected, 0, 1)
    
    return corrected


def load_correction_images(
    flatfield_path: Optional[Union[str, Path]] = None,
    dark_current_path: Optional[Union[str, Path]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load flatfield and dark current correction images.
    
    Args:
        flatfield_path: Path to flatfield image
        dark_current_path: Path to dark current image
        
    Returns:
        Tuple of (flatfield image, dark current image)
    """
    flatfield = None
    dark_current = None
    
    if flatfield_path is not None:
        flatfield = imageio.imread(flatfield_path)
        # Normalize to [0, 1] range
        flatfield = flatfield.astype(np.float32) / np.max(flatfield)
    
    if dark_current_path is not None:
        dark_current = imageio.imread(dark_current_path)
        # Normalize to [0, 1] range
        dark_current = dark_current.astype(np.float32) / np.max(dark_current)
    
    return flatfield, dark_current


def apply_corrections(
    image: np.ndarray,
    flatfield: Optional[np.ndarray] = None,
    dark_current: Optional[np.ndarray] = None,
    white_balance_method: str = "gray_world",
    reference_white: Optional[np.ndarray] = None,
    illumination_correction: bool = False,
    illumination_method: str = "polynomial"
) -> np.ndarray:
    """
    Apply a series of image corrections.
    
    Args:
        image: Input image array (height, width, channels)
        flatfield: Optional flatfield image array
        dark_current: Optional dark current image array
        white_balance_method: White balancing method
        reference_white: Optional reference white values for manual white balancing
        illumination_correction: Whether to apply non-uniform illumination correction
        illumination_method: Method for illumination correction
        
    Returns:
        Corrected image array
    """
    # Convert to float for calculations
    corrected = image.astype(np.float32)
    
    # Apply flatfield correction if flatfield is provided
    if flatfield is not None:
        corrected = flatfield_correction(corrected, flatfield, dark_current)
    
    # Apply white balancing
    corrected = white_balance(corrected, reference_white, white_balance_method)
    
    # Apply non-uniform illumination correction if requested
    if illumination_correction:
        corrected = non_uniform_illumination_correction(
            corrected, method=illumination_method
        )
    
    return corrected 