import numpy as np
import pytest
from pathlib import Path
from ppm_processor import PolychromaticPolarizationProcessor, ProcessingParameters, ImageData, PolarizationPair

@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    return np.random.rand(100, 100, 3)

@pytest.fixture
def processor():
    """Create a processor instance with default settings."""
    return PolychromaticPolarizationProcessor()

@pytest.fixture
def custom_processor():
    """Create a processor instance with custom settings."""
    params = ProcessingParameters(gain=0.7)
    return PolychromaticPolarizationProcessor(params)

def test_processor_initialization():
    """Test processor initialization with default and custom parameters."""
    # Default parameters
    processor = PolychromaticPolarizationProcessor()
    assert processor.params.gain == 0.6
    assert processor.params.window_shape == (16, 16, 3)
    assert processor.params.step == 4

    # Custom parameters
    params = ProcessingParameters(gain=0.7)
    processor = PolychromaticPolarizationProcessor(params)
    assert processor.params.gain == 0.7

def test_estimate_background(processor, sample_image_data):
    """Test background estimation function."""
    bg_mean, indices = processor.estimate_background(sample_image_data)
    
    # Check output types and shapes
    assert isinstance(bg_mean, np.ndarray)
    assert bg_mean.shape == (3,)  # 3 channels
    assert isinstance(indices, np.ndarray)
    assert len(indices) > 0

def test_process_background(processor, sample_image_data):
    """Test background processing function."""
    # Create a polarization pair
    pos_img = ImageData(data=sample_image_data, path=Path("test.tif"), name="test")
    neg_img = ImageData(data=sample_image_data, path=Path("test.tif"), name="test")
    image_pair = PolarizationPair(positive=pos_img, negative=neg_img)
    
    # Process background
    bg_pos_mean, bg_neg_mean, scale_pos, scale_neg = processor.process_background(image_pair)
    
    # Check outputs
    assert isinstance(bg_pos_mean, np.ndarray)
    assert isinstance(bg_neg_mean, np.ndarray)
    assert isinstance(scale_pos, float)
    assert isinstance(scale_neg, float)
    assert scale_pos > 0
    assert scale_neg > 0

def test_correct_images(processor, sample_image_data):
    """Test image correction function."""
    # Create a polarization pair
    pos_img = ImageData(data=sample_image_data, path=Path("test.tif"), name="test")
    neg_img = ImageData(data=sample_image_data, path=Path("test.tif"), name="test")
    image_pair = PolarizationPair(positive=pos_img, negative=neg_img)
    
    # Process background first
    bg_pos_mean, bg_neg_mean, scale_pos, scale_neg = processor.process_background(image_pair)
    
    # Correct images
    pos_neg_diff, neg_pos_diff, combined = processor.correct_images(
        image_pair, bg_pos_mean, bg_neg_mean, scale_pos, scale_neg
    )
    
    # Check outputs
    assert isinstance(pos_neg_diff, np.ndarray)
    assert isinstance(neg_pos_diff, np.ndarray)
    assert isinstance(combined, np.ndarray)
    assert pos_neg_diff.shape == sample_image_data.shape
    assert neg_pos_diff.shape == sample_image_data.shape
    assert combined.shape == sample_image_data.shape
    assert np.all(pos_neg_diff >= 0) and np.all(pos_neg_diff <= 1)
    assert np.all(neg_pos_diff >= 0) and np.all(neg_pos_diff <= 1)
    assert np.all(combined >= 0) and np.all(combined <= 1)

def test_invalid_inputs(processor):
    """Test error handling for invalid inputs."""
    # Test with None input
    with pytest.raises(ValueError):
        processor.estimate_background(None)
    
    # Test with wrong dimensions
    wrong_dim_data = np.random.rand(100, 100)  # 2D instead of 3D
    with pytest.raises(ValueError):
        processor.estimate_background(wrong_dim_data)
    
    # Test with empty array
    empty_data = np.array([])
    with pytest.raises(ValueError):
        processor.estimate_background(empty_data)

def test_custom_parameters(custom_processor, sample_image_data):
    """Test processor with custom parameters."""
    # Create a polarization pair
    pos_img = ImageData(data=sample_image_data, path=Path("test.tif"), name="test")
    neg_img = ImageData(data=sample_image_data, path=Path("test.tif"), name="test")
    image_pair = PolarizationPair(positive=pos_img, negative=neg_img)
    
    # Process with custom gain
    bg_pos_mean, bg_neg_mean, scale_pos, scale_neg = custom_processor.process_background(image_pair)
    pos_neg_diff, neg_pos_diff, combined = custom_processor.correct_images(
        image_pair, bg_pos_mean, bg_neg_mean, scale_pos, scale_neg
    )
    
    # Verify custom gain was used
    assert custom_processor.params.gain == 0.7 