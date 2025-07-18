# AUV Vision

This repository contains code and data for an Autonomous Underwater Vehicle (AUV) vision system, including training, testing, and model conversion scripts.

## Project Structure

```
Data/
  train/
    Distance/      # Training distance labels
    Mask/          # Training mask images
    Original/      # Original training images
  test/
    Distance/      # Test distance labels
    Mask/          # Test mask images
    Original/      # Original test images
Model/
  fast_scnn_pipe_traced.pt
  fast_scnn_pipe.mlmodel
  fast_scnn_pipe.onnx
  fast_scnn_pipe.pth
  light_distance_net.pth
test/
  distance_with_plot.py
  normal_test.py
  real_time_test.py
  test_distance.py
  test_line.py
  test.png
train/
  coreML.py
  gray_scale_converter.py
  to_onnx.py
  train_distance.py
  train_line.py
```

## Getting Started

1. **Install dependencies**  
   Make sure you have Python and required libraries installed.

2. **Training**  
   Use scripts in `train/` to train models:
   - `train_distance.py`: Train distance estimation model
   - `train_line.py`: Train line detection model

3. **Testing**  
   Use scripts in `test/` to evaluate models:
   - `test_distance.py`: Test distance estimation
   - `test_line.py`: Test line detection
   - `real_time_test.py`: Real-time inference

4. **Model Conversion**  
   Use `to_onnx.py` and `coreML.py` for converting models to ONNX and CoreML formats.

## Data Format

- **Distance labels**: Text files with a single float value.
- **Masks**: PNG images for segmentation.
- **Original images**: Raw input images.

## Models

- Pretrained models are stored in the `Model/` directory.
