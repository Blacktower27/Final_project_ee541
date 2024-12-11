# Final-Project-EE541

## Super-Resolution with Different Feature Extractors

This project is built upon SRGAN (Super-Resolution Generative Adversarial Network), with a focus on exploring the effects of different feature extractors in the generator. By replacing the feature extractor in the generator, we aim to analyze the impact on super-resolution performance.

---

## 1. Data

We use the DIV2K dataset for training and evaluation. It includes:

- **1600 Training Images:**
    - 800 High-Resolution (HR) images (2K)
    - 800 Low-Resolution (LR) images (4x downscaled bicubic)

- **200 Test Images:**
    - 100 HR
    - 100 LR

Data files are structured as follows:

```
DIV2K/
├── DIV2K_train_HR/                # 800 HR training images
├── DIV2K_train_LR_bicubic/X4/     # 800 LR training images (4x downscaled)
├── DIV2K_valid_HR/                # 100 HR validation images
└── DIV2K_valid_LR_bicubic/X4/     # 100 LR validation images (4x downscaled)
```

To use the dataset, download the DIV2K HR and bicubic LR (X2 and X4) images from the official [DIV2K dataset page](https://data.vision.ee.ethz.ch/cvl/DIV2K/), and extract them into the `DIV2K/` folder as shown above.

---

## 2. Repository Structure

```
.
├── CONFIG.py                      # Configurations for data, models, and training
├── DIV2K/                         # Contains the dataset
├── logs/                          # Logs for training with different feature extractors
├── models/                        # Saved models
├── results/                       # Results for validation images
├── src/                           # Source code for training, evaluation, and utilities
│   ├── Cnn_Block.py               # CNN-based feature extractor
│   ├── Dense_Block.py             # DenseNet-based feature extractor
│   ├── SwinTransformer_Block.py   # Swin Transformer feature extractor
│   ├── dataset.py                 # Custom PyTorch dataset for DIV2K
│   ├── discriminator.py           # Discriminator for GAN
│   ├── evaluate.py                # Evaluation script
│   ├── generator.py               # Generator for GAN
│   ├── loss.py                    # Loss functions
│   ├── train.py                   # Training script
│   ├── gen_sr.py                  # Inference script to generate SR images
│   ├── visualize.ipynb            # Visualization notebook for traning logs
│   └── eval_bilinear.py           # Baseline bilinear evaluation
├── unit_tests/                    # Unit tests for key components
└── README.md                      # Project description (this file)
```

---

## 3. Feature Extractors in Generator

We have implemented multiple feature extractors to replace the default one in SRGAN:

- **Residual_Block.py:** ResidualNet-based feature extraction.
- **Cnn_Block.py:** CNN-based feature extraction.
- **Dense_Block.py:** DenseNet-based feature extraction.
- **SwinTransformer_Block.py:** Swin Transformer-based feature extraction.

Each of these extractors is used in the generator to analyze its impact on super-resolution performance.

---

## 4. Usage

### Environment Setup

1. Clone the repository and navigate to the project directory.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Training

To train the model with a specific feature extractor, modify `CONFIG.py` to select the desired feature extractor, then run:

```bash
python src/train.py
```

### Generate Super-Resolution Images

Use the following command to generate super-resolution images for validation data:

```bash
python src/gen_sr.py
```

### Evaluate Performance

Run the evaluation script to compute PSNR and SSIM metrics for validation images:

```bash
python src/evaluate.py
```

---

## 5. Logs and Results

- **Training Logs:**
  Logs for each feature extractor can be found in the `logs/` directory. Example:
  - `training_log_cnn_X2.txt`
  - `training_log_dense_X2.txt`
  - `training_log_swinIR_X4.txt`

- **Validation Results:**
  Super-resolution outputs for validation images are saved in the `results/` directory. Example:
  - `results/sr_val_x2/`
  - `results/sr_val_x4/`
