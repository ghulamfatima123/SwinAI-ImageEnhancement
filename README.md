---

# SwinIR-Based Image Enhancement  

## Overview  
This project leverages the SwinIR model for various image enhancement tasks, including super-resolution (SR), denoising, and compression artifact reduction (CAR). The implementation dynamically loads the appropriate SwinIR model based on the specified checkpoint and processes images accordingly.  

## Features  
- Supports multiple SwinIR variants:  
  - **Classical Super-Resolution (SR)** (x2, x3, x4, x8)  
  - **Lightweight SR** (optimized for efficiency)  
  - **Real-World SR** (handles real-world images effectively)  
  - **Denoising (Gray/Color)**  
  - **Compression Artifact Reduction (CAR)**  
- Automatically detects model type from the checkpoint path.  
- Handles images of any size with automatic padding.  
- Efficient image processing using PyTorch.  

## Installation  
### Prerequisites  
Ensure you have Python 3.8+ and install the necessary dependencies:  
```bash  
pip install torch torchvision pillow numpy  
```  

## Usage  
### Image Enhancement  
Run the script to process an image using a specified SwinIR model:  
```bash  
python main.py
```

## Code Structure  
- `main.py`: Main script for loading the model and processing images.  
- `models/network_swinir.py`: Defines the SwinIR architecture.  
- `input_images/`: Folder containing images to be processed.  
- `output_images : where enhanced images are saved.  

## License  
This project is distributed under the **MIT License**. Ensure compliance with the original SwinIR authors' licensing terms if using their models.  

## Acknowledgments  
This project is based on [SwinIR](https://github.com/JingyunLiang/SwinIR) by Jingyun Liang et al. Please refer to their repository for further model details.  

---  
**Disclaimer:** This repository does not modify SwinIR's core architecture but provides a streamlined way to use its models for different image enhancement tasks.  

---
