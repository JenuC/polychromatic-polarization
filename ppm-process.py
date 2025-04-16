import numpy as np
from skimage import io, color, img_as_float, img_as_ubyte, exposure
from skimage.util.shape import view_as_windows
import os
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import tifffile
import sys

def estimate_background(img: np.ndarray, 
                       preset_indices: Optional[np.ndarray] = None,
                       window_shape: Tuple[int, int, int] = (16, 16, 3),
                       step: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate background from image by analyzing dark regions.
    
    Args:
        img: Input image array
        preset_indices: Optional pre-calculated indices for background regions
        window_shape: Shape of windows to analyze (default: 16x16x3)
        step: Step size for window sliding (default: 4)
        
    Returns:
        Tuple of (background mean values, indices of background regions)
    """
    img_windows = view_as_windows(img, window_shape, step)
    img_windows_flat = np.reshape(img_windows, 
                                (img_windows.shape[0] * img_windows.shape[1],
                                 window_shape[0], window_shape[1], window_shape[2]))
    
    if preset_indices is None:
        s_windows = np.sum(img_windows_flat, axis=(1, 2, 3))
        indices = np.argsort(s_windows)  # ascending order
        a = int(img_windows_flat.shape[0] * 0.0001)  # 0.01% of windows
        low_indices = indices[-a:]
    else:
        low_indices = preset_indices
        
    low_patches = img_windows_flat[low_indices]
    bg_mean = np.mean(low_patches, axis=(0, 1, 2))
    
    return bg_mean, low_indices

@dataclass
class ImageData:
    """Container for image data and its metadata"""
    data: np.ndarray
    path: Path
    name: str

@dataclass
class PolarizationPair:
    """Container for paired polarization images"""
    positive: ImageData
    negative: ImageData
    brightfield: Optional[ImageData] = None

@dataclass
class ProcessingParameters:
    """Container for image processing parameters"""
    gain: float = 0.6
    window_shape: Tuple[int, int, int] = (16, 16, 3)
    step: int = 4

class PolychromaticPolarizationProcessor:
    def __init__(self, params: ProcessingParameters = None):
        self.params = params or ProcessingParameters()
        self.base_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        
    def set_directory(self, directory: str) -> Dict[str, Path]:
        """Set the working directory and create necessary paths."""
        self.base_path = Path(directory).resolve()
        if not self.base_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
            
        self.output_path = self.base_path / 'results'
        self.output_path.mkdir(exist_ok=True)
        
        return {
            'background': self.base_path / 'bg',
            'positive': self.base_path / '+5',
            'negative': self.base_path / '-5',
            'brightfield': self.base_path / 'bf'
        }
        
    def get_sorted_files(self, paths: Dict[str, Path]) -> Dict[str, List[Path]]:
        files = {
            'positive': sorted(paths['positive'].glob('*')),
            'negative': sorted(paths['negative'].glob('*')),
            'brightfield': sorted(paths['brightfield'].glob('*'))
        }
        
        if not files['positive'] or not files['negative'] or len(files['negative']) != len(files['positive']):
            raise ValueError('Mismatched positive and negative image pairs')
            
        return files
        
    def load_image_pair(self, file_paths: Dict[str, Path], bg_path: Optional[Path] = None) -> PolarizationPair:
        if bg_path and bg_path.exists():
            pos_img = ImageData(
                data=img_as_float(io.imread(bg_path / 'b+5.tif')),
                path=bg_path / 'b+5.tif',
                name='background_positive'
            )
            neg_img = ImageData(
                data=img_as_float(io.imread(bg_path / 'b-5.tif')),
                path=bg_path / 'b-5.tif',
                name='background_negative'
            )
        else:
            pos_img = ImageData(
                data=img_as_float(io.imread(file_paths['positive'])),
                path=file_paths['positive'],
                name=file_paths['positive'].stem
            )
            neg_img = ImageData(
                data=img_as_float(io.imread(file_paths['negative'])),
                path=file_paths['negative'],
                name=file_paths['negative'].stem
            )
            
        return PolarizationPair(positive=pos_img, negative=neg_img)
        
    def process_background(self, image_pair: PolarizationPair) -> Tuple[np.ndarray, np.ndarray, float, float]:
        pos_sum = np.sum(image_pair.positive.data)
        neg_sum = np.sum(image_pair.negative.data)
        
        if pos_sum > neg_sum:
            bg_pos_mean, indices = estimate_background(
                image_pair.positive.data, 
                window_shape=self.params.window_shape,
                step=self.params.step
            )
            bg_neg_mean, _ = estimate_background(
                image_pair.negative.data,
                preset_indices=indices,
                window_shape=self.params.window_shape,
                step=self.params.step
            )
        else:
            bg_neg_mean, indices = estimate_background(
                image_pair.negative.data,
                window_shape=self.params.window_shape,
                step=self.params.step
            )
            bg_pos_mean, _ = estimate_background(
                image_pair.positive.data,
                preset_indices=indices,
                window_shape=self.params.window_shape,
                step=self.params.step
            )
            
        bg_pos_max = np.max(bg_pos_mean)
        bg_neg_max = np.max(bg_neg_mean)
        
        scale_pos = max(bg_pos_max, bg_neg_max) / bg_pos_max
        scale_neg = max(bg_pos_max, bg_neg_max) / bg_neg_max
        
        return (
            bg_pos_max / bg_pos_mean,
            bg_neg_max / bg_neg_mean,
            scale_pos,
            scale_neg
        )
        
    def correct_images(self, 
                    image_pair: PolarizationPair,
                    bg_pos_mean: np.ndarray,
                    bg_neg_mean: np.ndarray,
                    scale_pos: float,
                    scale_neg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        corrected_pos = np.clip(image_pair.positive.data * bg_pos_mean * scale_pos, 0, 1)
        corrected_neg = np.clip(image_pair.negative.data * bg_neg_mean * scale_neg, 0, 1)
        
        pos_mean = np.mean(image_pair.positive.data)
        neg_mean = np.mean(image_pair.negative.data)
        
        intensity_range = (0.1 - 0.1*self.params.gain, 1-self.params.gain)
        
        if neg_mean > pos_mean:
            pos_neg_diff = exposure.rescale_intensity(
                np.clip(corrected_pos - corrected_neg, 0, 1),
                in_range=intensity_range,
                out_range=(0, 1)
            )
            neg_pos_diff = exposure.rescale_intensity(
                np.clip(corrected_neg - corrected_pos, 0, 1),
                in_range=intensity_range,
                out_range=(0, 1)
            )
        else:
            neg_pos_diff = exposure.rescale_intensity(
                np.clip(corrected_neg - corrected_pos, 0, 1),
                in_range=intensity_range,
                out_range=(0, 1)
            )
            pos_neg_diff = exposure.rescale_intensity(
                np.clip(corrected_pos - corrected_neg, 0, 1),
                in_range=intensity_range,
                out_range=(0, 1)
            )
            
        return pos_neg_diff, neg_pos_diff, np.clip(pos_neg_diff + neg_pos_diff, 0, 1)
        
    def save_results(self, 
                    image_pair: PolarizationPair,
                    pos_neg_diff: np.ndarray,
                    neg_pos_diff: np.ndarray,
                    combined_result: np.ndarray,
                    display: bool = False):
        base_name = image_pair.positive.name
        
        tifffile.imwrite(str(self.output_path / f"{base_name}-5.tif"), img_as_ubyte(neg_pos_diff))
        tifffile.imwrite(str(self.output_path / f"{base_name}+5.tif"), img_as_ubyte(pos_neg_diff))
        tifffile.imwrite(str(self.output_path / f"{base_name}_color.tif"), img_as_ubyte(combined_result))
        
        gray = np.amax(combined_result, 2)
        tifffile.imwrite(str(self.output_path / f"{base_name}_gray.tif"), img_as_ubyte(gray))
        
        if image_pair.brightfield is not None:
            green = np.zeros(image_pair.brightfield.data.shape)
            green[:, :, 1] = gray
            overlay = np.clip(image_pair.brightfield.data + green, 0, 1)
            tifffile.imwrite(str(self.output_path / f"{base_name}_overlay.tif"), img_as_ubyte(overlay))
            
            # Create and save the figure
            plt.figure(figsize=(10, 3))
            plt.subplot(121).set_title("Result image")
            plt.imshow(combined_result)
            plt.axis('off')
            plt.subplot(122).set_title("Overlay image")
            plt.imshow(overlay)
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(str(self.base_path / f"{base_name}_panels.png"), dpi=75, bbox_inches='tight')
            
            # Display only if requested
            if display:
                plt.show()
            else:
                plt.close()
            
    def process_images(self, directory: str, display: bool = False):
        paths = self.set_directory(directory)
        files = self.get_sorted_files(paths)
        
        for pos_path, neg_path, bf_path in zip(files['positive'], files['negative'], files['brightfield']):
            image_pair = self.load_image_pair({
                'positive': pos_path,
                'negative': neg_path
            }, paths['background'])
            
            if bf_path.exists():
                image_pair.brightfield = ImageData(
                    data=img_as_float(io.imread(bf_path)),
                    path=bf_path,
                    name=bf_path.stem
                )
            
            print(f"Processing: {image_pair.positive.name}")
            
            bg_pos_mean, bg_neg_mean, scale_pos, scale_neg = self.process_background(image_pair)
            pos_neg_diff, neg_pos_diff, combined_result = self.correct_images(
                image_pair, bg_pos_mean, bg_neg_mean, scale_pos, scale_neg
            )
            
            self.save_results(image_pair, pos_neg_diff, neg_pos_diff, combined_result, display)
            print(f"Result saved for: {image_pair.positive.name}")

def main():
    parser = argparse.ArgumentParser(
        description='Process polychromatic polarization microscopy images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images in the data-sample directory with default gain
  python ppm-process.py data-sample

  # Process images with custom gain parameter
  python ppm-process.py data-sample --gain 0.7

  # Process images and display results
  python ppm-process.py data-sample --display

Directory structure should be:
  data-sample/
    ├── +5/         # Positive polarization images
    ├── -5/         # Negative polarization images
    ├── bf/         # Brightfield images (optional)
    └── bg/         # Background images (optional)
        ├── b+5.tif
        └── b-5.tif
"""
    )
    parser.add_argument('directory', type=str, help='Directory containing the image data')
    parser.add_argument('--gain', type=float, default=0.6, 
                       help='Image gain parameter (default: 0.6, range: 0.0-1.0)')
    parser.add_argument('--display', action='store_true',
                       help='Display results while processing')
    
    try:
        args = parser.parse_args()
        
        # Validate gain parameter
        if not 0.0 <= args.gain <= 1.0:
            parser.error("Gain must be between 0.0 and 1.0")
            
        params = ProcessingParameters(gain=args.gain)
        processor = PolychromaticPolarizationProcessor(params)
        processor.process_images(args.directory, args.display)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
