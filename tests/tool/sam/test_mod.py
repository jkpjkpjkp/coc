import pytest
import numpy as np
from PIL import Image
import threading
from coc.tool.sam.automagen import SAMWrapper, get_sam

# Session-scoped fixture to initialize SAMWrapper once
@pytest.fixture(scope="session")
def sam_processor():
    """Fixture to provide a SAM processor with variant='t' and max_parallel=1."""
    processor = get_sam(
        variant='t',
        max_parallel=1,
    )
    return processor

# Fixture to load a sample image
@pytest.fixture
def sample_image():
    """Fixture to load onions.jpg as a NumPy array."""
    image_path = "data/sample/onions.jpg"
    image = Image.open(image_path)
    return np.array(image)

### Test 1: Initialization
def test_initialization(sam_processor):
    """Test that SAMWrapper initializes correctly and loads the model."""
    assert sam_processor is not None
    assert sam_processor._sam_engine.model is not None, "Model should be loaded during initialization"

### Test 2: Mask Generation
def test_generate_masks(sam_processor, sample_image):
    """Test that generate_masks produces a dictionary output."""
    mask_data = sam_processor._sam_engine.generate_masks(sample_image)
    assert isinstance(mask_data, dict), "generate_masks should return a dictionary"
    assert len(mask_data) > 0, "Mask data should not be empty"

### Test 3: Post-Processing
def test_post_process_masks(sam_processor, sample_image):
    """Test that post_process_masks returns a list of dictionaries."""
    mask_data = sam_processor._sam_engine.generate_masks(sample_image)
    annotations = sam_processor._sam_engine.post_process_masks(mask_data)
    assert isinstance(annotations, list), "post_process_masks should return a list"
    assert len(annotations) > 0, "Annotations list should not be empty"
    assert isinstance(annotations[0], dict), "Each annotation should be a dictionary"

### Test 4: Visualization (Assuming show_anns returns img)
def test_visualize_annotations(sam_processor, sample_image):
    """Test that visualize_annotations returns a NumPy array with the correct shape."""
    annotations = sam_processor(sample_image)
    vis_image = sam_processor._sam_engine.visualize_annotations(sample_image, annotations)
    assert isinstance(vis_image, np.ndarray), "visualize_annotations should return a NumPy array"
    # Assuming the output has 4 channels (RGBA) and matches input height/width
    assert vis_image.shape[:2] == sample_image.shape[:2], "Visualization dimensions should match input image"
    assert vis_image.shape[2] == 4, "Visualization should have 4 channels (RGBA)"

### Test 5: Concurrency Control
def test_concurrency():
    """Test that the semaphore limits concurrent processing to max_parallel."""
    # Initialize with max_parallel=2 for this test
    sam_processor = get_sam(
        variant='t',
        max_parallel=2,
        checkpoint_dir="/home/jkp/Pictures/sam2/checkpoints"
    )

    # Load sample images
    image_paths = ["data/sample/onions.jpg", "data/sample/4girls.jpg"]
    images = [np.array(Image.open(path)) for path in image_paths] * 2  # Duplicate to get 4 images

    # Thread-safe counter for concurrent processes
    counter = 0
    max_counter = 0
    counter_lock = threading.Lock()

    def process_image(image):
        nonlocal counter, max_counter
        with counter_lock:
            counter += 1
            if counter > max_counter:
                max_counter = counter
        sam_processor(image)
        with counter_lock:
            counter -= 1

    # Create and start 4 threads
    threads = [threading.Thread(target=process_image, args=(image,)) for image in images]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_counter <= 2, f"Maximum concurrent processes ({max_counter}) should not exceed max_parallel (2)"

### Test 6: Parameter Override
def test_parameter_override(sam_processor, sample_image):
    """Test that custom parameters can be passed without errors."""
    custom_params = {'points_per_side': 16}
    annotations = sam_processor(sample_image, **custom_params)
    assert isinstance(annotations, list), "Annotations should be a list with custom parameters"
    assert len(annotations) > 0, "Annotations list should not be empty with custom parameters"

### Test 7: Error Handling
def test_invalid_input(sam_processor):
    """Test that passing invalid input raises an appropriate exception."""
    with pytest.raises(TypeError):
        sam_processor("not an image")  # Should raise TypeError due to invalid input type