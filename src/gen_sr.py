import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms as trf
from PIL import Image
from generator import Generator
from train import PATH_G, xavier_init_weights, transform_lr
from CONFIG import *
from glob import glob

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories and paths
base_dir = Path(__file__).resolve().parent.parent
SR_VAL_DIR = base_dir / "results" / f"sr_val_x{UPSCALE}"
div2k_valid_lr_dir = base_dir / "DIV2K" / f"DIV2K_valid_LR_bicubic" / f"X{UPSCALE}"

# Ensure directories exist
SR_VAL_DIR.mkdir(parents=True, exist_ok=True)

def tensor_to_img(tensor, filepath):
    tensor = tensor.cpu()
    pil = trf.ToPILImage()(tensor.squeeze_(0))
    pil.save(filepath)
    print(f"Saved to {filepath}")

def gen_sr_valset():
    files = glob(str(div2k_valid_lr_dir / "*.png"))
    for lr_img_path in files:
        with torch.no_grad():
            pil_img = Image.open(lr_img_path)
            img_tensor = trf.ToTensor()(pil_img)
            img_tensor = torch.unsqueeze(img_tensor, 0)  # Add batch dimension
            img_tensor = img_tensor.to(device)
            sr_img = G(img_tensor)

        file_name = Path(lr_img_path).name
        sr_img_path = SR_VAL_DIR / f"sr_{file_name}"
        tensor_to_img(sr_img, sr_img_path)

if __name__ == '__main__':
    # Load generator model
    G = Generator(n_blks=N_BLK_G, upscale_factor=UPSCALE)
    if PATH_G.exists():
        checkpoint_G = torch.load(PATH_G)
        G.load_state_dict(checkpoint_G['state_dict'])
        G.to(device)
    else:
        print("Checkpoints not found, using Xavier initialization.")
        G.apply(xavier_init_weights).to(device)
    G.eval()

    # Generate super-resolution images
    gen_sr_valset()
