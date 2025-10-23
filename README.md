# A Comparative Study of CNN-LSTM and CNN-ViT Architectures for Detecting AI-Generated Videos

This repository contains the official PyTorch implementation for the final year project titled "A Comparative Study of CNN-LSTM and CNN-ViT Architectures for Detecting AI-Generated Videos."

## Overview

This project investigates and benchmarks a novel hybrid deep learning framework for the detection of fully AI-generated videos. The core of this research is a rigorous, controlled comparison between two architectural variants:

1.  **CNN-LSTM:** A recurrent approach combining a CNN backbone with a Long Short-Term Memory (LSTM) network to model local, sequential temporal artifacts.
2.  **CNN-ViT:** An attention-based approach that replaces the LSTM with a Vision Transformer (ViT) encoder to capture global, long-range dependencies across video frames.

Both models are augmented with motion features derived from optical flow to enhance the detection of physical and temporal inconsistencies. Our findings demonstrate that the **CNN-ViT architecture is significantly more robust and accurate**, establishing it as a state-of-the-art solution for this task.

## Key Features

- **Hybrid Framework:** Combines a pre-trained ResNet-18 for spatial feature extraction with a dedicated temporal backbone.
- **Multi-Modal Input:** Utilizes both RGB frames (spatial) and pre-computed optical flow features (motion).
- **Two Architectural Variants:** Provides complete, end-to-end training and evaluation scripts for both CNN-LSTM and CNN-ViT.
- **State-of-the-Art Performance:** The CNN-ViT model achieves **99.42%** accuracy on in-distribution test data and **91.60%** on a challenging cross-dataset generalization test.
- **Single Video Inference:** Includes standalone scripts to run inference on a single uploaded video file, with on-the-fly optical flow computation.

## Performance Highlights

### In-Distribution Test Set Results

| Metric | CNN-LSTM | CNN-ViT |
| :--- | :--- | :--- |
| **Overall Accuracy** | 97.13% | **99.42%** |
| **Weighted F1-Score** | 0.97 | **0.99** |
| Precision (Fake Class) | 0.98 | **0.99** |
| Recall (Fake Class) | 0.96 | **1.00** |

### Cross-Dataset Generalization Test (on Unseen Generators)

| Metric | CNN-LSTM | CNN-ViT |
| :--- | :--- | :--- |
| **Overall Accuracy** | 84.00% | **91.60%** |
| **Weighted F1-Score**| 0.84 | **0.92** |
| Recall (Fake Class) | 0.72 | **0.85** |

## Setup and Installation

This project was developed using Python 3.10 and PyTorch 2.2 in a Google Colab environment with an NVIDIA L4 GPU.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JTZY09/AI-Generated-Video-Detection-using-Hybrid-Architectures/tree/main
    cd your-repo-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.

## Data Preparation

The training scripts expect data to be pre-processed and organized as follows:

1.  **Frame Extraction:** Videos are converted into sequences of frames.
2.  **Optical Flow Pre-computation:** For each sequence, optical flow features are calculated using the Gunnar Farnebäck algorithm and saved as `.pt` files. The `preprocess/compute_flow.py` script can be used for this purpose.

*Note: Due to the large size of the datasets, it is recommended to prepare the data, zip it, and upload it to a cloud service like Google Drive. The training scripts are designed to work in an environment like Google Colab where this data is copied to a local runtime for performance.*

## Usage

### Training and Evaluation

To train and evaluate a model, run the corresponding script. Ensure the paths to your local data are correctly set within the script.

**Train the CNN-ViT model:**
```bash
python train_cnn_vit.py

**Train the CNN-LSTM model:**
python train_cnn_lstm.py
