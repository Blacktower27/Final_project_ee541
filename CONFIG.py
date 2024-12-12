from pathlib import Path

LR_CROPPED_SIZE = 100
UPSCALE = 4
HR_CROPPED_SIZE = UPSCALE * LR_CROPPED_SIZE

base_dir = Path(__file__).resolve().parent

TRAIN_HR_DIR = base_dir / "DIV2K" / "DIV2K_train_HR"
TRAIN_LR_DIR = base_dir / "DIV2K" / f"DIV2K_train_LR_bicubic" / f"X{UPSCALE}"
VAL_HR_DIR = base_dir / "DIV2K" / "DIV2K_valid_HR"
VAL_LR_DIR = base_dir / "DIV2K" / f"DIV2K_valid_LR_bicubic" / f"X{UPSCALE}"

REAL_VALUE = 0.99
FAKE_VALUE = 0.0

BATCH_SIZE = 8
EPOCHS = 50
WARMUP_EPOCHS = 50

N_BLK_G = 20
LR = 0.0001
BETAS = (0.5, 0.9)

VGG_LOSS_COEF = 0.006
ADVERSARIAL_LOSS_COEF = 0.001

# Specify feature extractor: 'Residual_Block', 'Cnn_Block', 'Dense_Block', or 'SwinTransformer_Block'
FEATURE_EXTRACTOR = "Residual_Block"