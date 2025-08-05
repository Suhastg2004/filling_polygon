# Polygon Color Filling with UNet

A Color-Conditioned UNet that fills polygon outlines with specified colors.


Prediction
<img width="1154" height="390" alt="Image" src="https://github.com/user-attachments/assets/4aeff13d-4333-40fb-8f4b-f5436b14096a" />



https://github.com/user-attachments/assets/62afdd37-8c89-4736-984a-0c6f33325129

## Overview

* **Input**: Polygon outline image + color name (text)
* **Output**: Filled polygon image
* **Framework**: PyTorch
* **Implementation**: Single Jupyter notebook
* **Tracking**: Weights & Biases integration

## Repository Structure

```
├── fill_polygon_unet_final.ipynb       # Complete implementation
├── dataset/                            # Training data
│   ├── training/
│   └── validation/
└── README.md
```

## Quick Start

```bash
git clone https://github.com/yourusername/polygon-color-filling-unet.git
cd polygon-color-filling-unet
pip install torch torchvision numpy pillow wandb matplotlib
jupyter notebook fill_polygon_unet_final.ipynb
```

## Usage

Run in the notebook:

```python
# Training
model, color_encoder = train_model_with_wandb()

# Testing
overall_acc, overall_iou = comprehensive_test_fixed('best_model.pth')
```

## Results

| Metric           | Value  |
| ---------------- | ------ |
| Final Train Loss | 0.1405 |
| Final Val Loss   | 0.1506 |
| Best Val Loss    | 0.1351 |
| Train Accuracy   | 99.98% |
| Val Accuracy     | 99.98% |
| Test Accuracy    | 62.48% |
| Test IoU         | 62.28% |
| Parameters       | 32M    |

## Supported Colors

`red, green, blue, yellow, purple, orange, pink, cyan, brown, black`

## Architecture

* **Color Embedding**: 64-dimensional learnable representations
* **Spatial Projection**: Color maps to 128×128 spatial dimensions
* **UNet Backbone**: Encoder-decoder with skip connections
* **Input Channels**: 4 (RGB + color conditioning)

## Experiment Tracking

View live dashboard: [Weights & Biases](https://api.wandb.ai/links/suhastg1282004-na/9a83o647)

## Contact

**Suhas TG**
Email: [suhastg1282004@gmail.com](mailto:suhastg1282004@gmail.com)
LinkedIn: [linkedin.com/in/suhastg]([https://linkedin.com/in/suhastg](https://www.linkedin.com/in/suhastg2004/))
