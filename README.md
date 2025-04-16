# Polychromatic polarization microscopy image processing
Polychromatic polarization microscopy is a real-time collagen imaging method in clinical histopathology compatible with brighfield microscopy.

|Brightfield| PPM |
|----------|--------|
|<img src="https://github.com/uw-loci/polychromatic-polarization/blob/master/thumbnails/brightfield.png" width="320">|<img src="https://github.com/uw-loci/polychromatic-polarization/blob/master/thumbnails/ppm.png" width="320">|


# PPP Process

A Python package for processing polychromatic polarization measurements, particularly effective for visualizing collagen structures in clinical histopathology.

## Installation

```bash
pip install ppp
```

## Usage

### Command Line Interface

Process images from a directory containing polarization data:

```bash
# Basic usage with default settings
ppp data-sample

# Process with custom gain parameter (default: 0.6)
ppp data-sample --gain 0.7

# Process and display results while processing
ppp data-sample --display
```

Directory structure should be:
```
data-sample/
├── +5/         # Positive polarization images
├── -5/         # Negative polarization images
├── bf/         # Brightfield images (optional)
└── bg/         # Background images (optional)
    ├── b+5.tif
    └── b-5.tif
```

### Python API

```python
from ppm_process import PolychromaticPolarizationProcessor
from pathlib import Path

# Create processor with default settings
processor = PolychromaticPolarizationProcessor()

# Process images from a directory
processor.process_images("data-sample", display=True)

# Or process with custom gain
from ppm_process import ProcessingParameters
params = ProcessingParameters(gain=0.7)
processor = PolychromaticPolarizationProcessor(params)
processor.process_images("data-sample")
```

The processor will:
1. Load and process the polarization image pairs
2. Generate PPM results
3. Save multiple output versions:
   - Individual +5 and -5 difference images
   - Combined result image
   - Color version
   - Grayscale version
   - Optional overlay with brightfield image

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

## Requirements

- Python >= 3.12
- numpy
- matplotlib
- scikit-image
- tifffile

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
