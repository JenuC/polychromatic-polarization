# Polychromatic polarization microscopy image processing
Polychromatic polarization microscopy is a real-time collagen imaging method in clinical histopathology compatible with brighfield microscopy.

|Brightfield| PPM |
|----------|--------|
|<img src="https://github.com/uw-loci/polychromatic-polarization/blob/master/thumbnails/brightfield.png" width="320">|<img src="https://github.com/uw-loci/polychromatic-polarization/blob/master/thumbnails/ppm.png" width="320">|


# PPM Process

A Python package for processing polychromatic polarization microscopy images.

## Installation

```bash
pip install ppm-process
```

## Usage

```python
from ppm_processor import PolychromaticPolarizationProcessor

# Create a processor instance
processor = PolychromaticPolarizationProcessor()

# Process images
processor.process_images(
    input_dir="path/to/input",
    output_dir="path/to/output",
    parameters=ProcessingParameters(
        wavelength=550,  # nm
        pixel_size=0.1,  # μm
        numerical_aperture=0.4
    )
)
```

## Features

- Process polychromatic polarization microscopy images
- Support for multiple image formats
- Configurable processing parameters
- Efficient batch processing
- Progress tracking and logging

## Requirements

- Python >= 3.10
- NumPy
- SciPy
- scikit-image
- imageio

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Technical Details

### Calculation Process

#### 1. Image Input and Background Estimation
- Takes two sets of images:
  - Positive (+5) polarization images
  - Negative (-5) polarization images
- For each image pair, it either:
  - Uses background images (`b+5.tif` and `b-5.tif`) if available
  - Or uses the actual images themselves for background estimation

#### 2. Background Correction Process
The code estimates background using the `estimate_background()` function which:
- Divides the image into small windows (16x16x3 pixels)
- Finds the darkest regions (lowest 0.01% of windows)
- Calculates the mean background value from these dark regions
- Applies background correction to both positive and negative images

#### 3. PPM Calculation Steps

##### a. Background Normalization
```python
bg_pos_max = np.max(bg_pos_mean)
bg_neg_max = np.max(bg_neg_mean)
a_pos = max(bg_pos_max, bg_neg_max) / bg_pos_max
a_neg = max(bg_pos_max, bg_neg_max) / bg_neg_max
```
- Normalizes the background values to ensure consistent scaling

##### b. Image Correction
```python
cor_img_pos = np.clip(img_pos * bg_pos_mean * a_pos, 0, 1)
cor_img_neg = np.clip(img_neg * bg_neg_mean * a_neg, 0, 1)
```
- Applies the background correction to both positive and negative images
- Clips values to ensure they stay in the valid range [0,1]

##### c. Polarization Difference Calculation
- Compares the mean values of corrected images to determine which difference to calculate
- Calculates two difference images:
```python
pos_neg = exposure.rescale_intensity(np.clip(cor_img_pos - cor_img_neg, 0, 1), 
                                   in_range=(0.05, 0.5), 
                                   out_range=(0, 1))
neg_pos = exposure.rescale_intensity(np.clip(cor_img_neg - cor_img_pos, 0, 1), 
                                   in_range=(0.05, 0.5), 
                                   out_range=(0, 1))
```
- The difference is rescaled to enhance the contrast in the meaningful range (0.05 to 0.5)

##### d. Final PPM Image
```python
result_img = np.clip(pos_neg + neg_pos, 0, 1)
```
- Combines both difference images to create the final PPM image
- The result represents the polarization-dependent differences in the sample

### Output Generation
The process saves multiple versions of the result:
- Individual +5 and -5 difference images
- Combined result image
- Color version
- Grayscale version
- Optional overlay with brightfield image
