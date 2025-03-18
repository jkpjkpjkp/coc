# Chain of Cognition - Enhanced Vision Capabilities

## Overview

This project enhances the capabilities of the GeminiAgent to handle complex visual reasoning tasks through a multi-step orchestrated approach. The implementation focuses on improving accuracy in tasks like counting objects and extracting pricing information from retail shelf images.

## Key Features

- **Orchestrated Vision Analysis**: Uses a multi-step approach with specialized vision tools to analyze images.
- **Layer-by-Layer Processing**: Divides shelf images into layers for more accurate counting and spatial reasoning.
- **Enhanced Evaluation Framework**: Updated evaluation tools for testing on the MUIR dataset with configurable options.
- **3D Vision Capabilities**: Integration with depth segmentation, novel view synthesis, and point cloud analysis.

## Usage

### Testing with an Image

```bash
python count_bottles.py path/to/image.jpg --verbose --save
```

### Running MUIR Evaluation

```bash
python run_muir_eval.py --with-3d --orchestration --verbose
```

### Comparing Approaches

```bash
python count_bottles.py path/to/image.jpg --compare
```

## Implementation Details

The `GeminiAgent` class in `coc/tree/gemi.py` provides:

- Standard generation with `generate()` - direct Gemini model usage
- Orchestrated generation with `generate_orchestrated()` - multi-step analysis with specialized tools

The orchestrated approach is particularly effective for:
- Counting objects in complex arrangements
- Analyzing retail shelves by layer
- Extracting pricing information from product displays

should we add RAG? (historically successful traj)



