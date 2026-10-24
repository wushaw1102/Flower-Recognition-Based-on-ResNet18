
---

# FlowersRecognition

![Python](https://img.shields.io/badge/Python-3.8-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.8.3-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A deep learning project for classifying 102 flower types using ResNet18 with PyTorch. Features data augmentation, two-stage training (feature extraction + fine-tuning), and visualizations of training results and predictions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [License](#license)

---

## Project Overview
*FlowersRecognition* is a deep learning project that uses a pre-trained ResNet18 model (from torchvision) to classify images of 102 flower species. The model is trained in two stages: first with feature extraction, then with fine-tuning, leveraging data augmentation to enhance performance.

---

## Features
- Classifies 102 flower categories using ResNet18.
- Data augmentation: random rotation, flips, color jitter, etc.
- Two-stage training: feature extraction followed by full fine-tuning.
- Visualizations: training/validation accuracy/loss plots, learning rate, and sample predictions.
- GPU support with PyTorch for faster training.

---

## Installation

### Prerequisites
- Python 3.8
- PyTorch with CUDA support (optional, for GPU training)
- Required libraries:
  ```bash
  pip install torch torchvision numpy pandas matplotlib pillow
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/wushaw1102/FlowersRecognition.git
   cd FlowersRecognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Place the dataset in the `./flower_data/` directory with `train` and `valid` subfolders (see [Dataset](#dataset) section).
2. Ensure `cat_to_name.json` is in the root directory for flower name mapping.
3. Run the main script:
   ```bash
   python main.py
   ```
4. Outputs:
   - Best model saved as `best.pt`.
   - Training results saved as `training_results_1.csv` and `training_results_2.csv`.
   - Plots saved as `training_results_plots_1.png` and `training_results_plots_2.png`.
   - Sample predictions saved as `flower_result.png`.

---

## Dataset
- **Source**: Flower dataset with 102 categories (not included due to size).
- **Structure**:
  - `./flower_data/train/`: Training images in subfolders named by class ID (e.g., `1`, `2`, ..., `102`).
  - `./flower_data/valid/`: Validation images in similar subfolders.
- **Mapping**: `cat_to_name.json` maps class IDs (e.g., "1") to flower names (e.g., "pink primrose").
- Download a compatible dataset (e.g., from [Kaggle's 102 Flower Classification](https://www.kaggle.com/datasets/aksha05/flower-images)) and organize it accordingly.

---

## Results
- **Best Validation Accuracy**: [填写你的最佳验证准确率，例如 0.92]
- **Training Stages**:
  - Stage 1 (Feature Extraction): Frozen layers, optimized fc layer.
  - Stage 2 (Fine-Tuning): All layers trainable with lower learning rate.
- Detailed metrics are in `training_results_1.csv` and `training_results_2.csv`.

---

## Visualizations
Outputs are saved in the root directory:
- **Training Plots (Stage 1)**: `training_results_plots_1.png`
- **Training Plots (Stage 2)**: `training_results_plots_2.png`
- **Sample Predictions**: `flower_result.png`

Example prediction:
![Sample Predictions](./flower_result.png)

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

