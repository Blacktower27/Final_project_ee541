import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
import torchvision.transforms as trf
from torch.utils.data import DataLoader
from torchsummary import summary

from CONFIG import *
from dataset import DIV2K
from discriminator import Discriminator
from generator import Generator
from loss import PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = Path(__file__).resolve().parent.parent
model_dir = base_dir / "models"
model_dir.mkdir(exist_ok=True)

log_dir = base_dir / "logs"
log_dir.mkdir(exist_ok=True)


model_name = FEATURE_EXTRACTOR.lower()
PATH_G = model_dir / f"{model_name}_G_X{UPSCALE}.pt"
PATH_D = model_dir / f"{model_name}_D_X{UPSCALE}.pt"
LOG_FILE = log_dir / f"training_log_{model_name}_X{UPSCALE}.txt"

transform_hr = trf.Compose([
    trf.CenterCrop(HR_CROPPED_SIZE),
    trf.ToTensor()
])
transform_lr = trf.Compose([
    trf.CenterCrop(LR_CROPPED_SIZE),
    trf.ToTensor()
])

def train(resume_training=True):
    if not resume_training and LOG_FILE.exists():
        LOG_FILE.unlink()

    data_train_hr, data_train_lr = load_training_data()
    hr_train_loader = DataLoader(dataset=data_train_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    lr_train_loader = DataLoader(dataset=data_train_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    assert len(hr_train_loader) == len(lr_train_loader)

    G = Generator(n_blks=N_BLK_G, upscale_factor=UPSCALE).to(device)
    D = Discriminator().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    if resume_training and PATH_G.exists() and PATH_D.exists():
        G, D, optimizer_G, optimizer_D, prev_epochs = load_checkpoints(G, D, optimizer_G, optimizer_D)
        warmup = False
    else:
        G.apply(xavier_init_weights)
        D.apply(xavier_init_weights)
        prev_epochs = 0
        summary(G, input_size=(3, LR_CROPPED_SIZE, LR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
        summary(D, input_size=(3, HR_CROPPED_SIZE, HR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
        warmup = True

    G.train()
    D.train()

    criterion_G = PerceptualLoss(vgg_coef=VGG_LOSS_COEF, adversarial_coef=ADVERSARIAL_LOSS_COEF).to(device)
    warmup_loss = torch.nn.L1Loss()
    criterion_D = torch.nn.BCELoss()

    if warmup:
        for w in range(WARMUP_EPOCHS):
            for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
                hr_img, lr_img = hr_batch[0].to(device), lr_batch[0].to(device)
                optimizer_G.zero_grad()

                sr_img = G(lr_img)
                err_G = warmup_loss(sr_img, hr_img)
                err_G.backward()

                optimizer_G.step()

                log_to_file(LOG_FILE, f"Warmup Epoch {w+1}, Batch {batch+1}: MAE Loss = {err_G.item():.4f}")
                if batch % 10 == 0:
                    print(f"\tBatch: {batch + 1}/{len(data_train_hr) // BATCH_SIZE}")
                    print(f"\tMAE G: {err_G.item():.4f}")

    for e in range(EPOCHS):
        for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
            hr_img, lr_img = hr_batch[0].to(device), lr_batch[0].to(device)

            optimizer_D.zero_grad()
            real_labels = torch.full(size=(len(hr_img),), fill_value=REAL_VALUE, dtype=torch.float, device=device)
            output_real = D(hr_img).view(-1)
            err_D_real = criterion_D(output_real, real_labels)
            err_D_real.backward()

            fake_labels = torch.full(size=(len(hr_img),), fill_value=FAKE_VALUE, dtype=torch.float, device=device)
            sr_img = G(lr_img)
            output_fake = D(sr_img.detach()).view(-1)
            err_D_fake = criterion_D(output_fake, fake_labels)
            err_D_fake.backward()

            optimizer_D.step()

            optimizer_G.zero_grad()
            output_fake = D(sr_img).view(-1)
            pixel_loss, adversarial_loss, vgg_loss = criterion_G(sr_img, hr_img, output_fake)
            err_G = pixel_loss + adversarial_loss + vgg_loss
            err_G.backward()
            optimizer_G.step()

            D_x = output_real.mean().item()
            D_Gz2 = output_fake.mean().item()
            log_message = (
                f"Epoch {e+prev_epochs+1}, Batch {batch+1}: "
                f"err_D_real = {err_D_real.item():.4f}, err_D_fake = {err_D_fake.item():.4f}, err_G = {err_G.item():.4f}, "
                f"D_x = {D_x:.4f}, D_Gz2 = {D_Gz2:.4f}, adversarial_loss = {adversarial_loss:.4f}, "
                f"vgg_loss = {vgg_loss:.4f}, pixel_loss = {pixel_loss:.4f}"
            )
            log_to_file(LOG_FILE, log_message)

            if batch % 10 == 0:
                print(log_message)

            del hr_img, lr_img, err_D_fake, err_D_real, err_G, real_labels, fake_labels, \
                output_real, output_fake, sr_img, pixel_loss, adversarial_loss, vgg_loss
            torch.cuda.empty_cache()

        save_checkpoints(G, D, optimizer_G, optimizer_D, epoch=prev_epochs+e+1)

def log_to_file(file_path, message):
    with open(file_path, "a") as f:
        f.write(message + "\n")

def load_training_data():
    data_train_hr = DIV2K(data_dir=base_dir / TRAIN_HR_DIR, transform=transform_hr)
    data_train_lr = DIV2K(data_dir=base_dir / TRAIN_LR_DIR, transform=transform_lr)
    return data_train_hr, data_train_lr

def save_checkpoints(G, D, optimizer_G, optimizer_D, epoch):
    checkpoint_G = {
        'model': G,
        'state_dict': G.state_dict(),
        'optimizer': optimizer_G.state_dict(),
        'epoch': epoch
    }
    checkpoint_D = {
        'model': D,
        'state_dict': D.state_dict(),
        'optimizer': optimizer_D.state_dict(),
    }
    torch.save(checkpoint_G, PATH_G)
    torch.save(checkpoint_D, PATH_D)

def load_checkpoints(G, D, optimizerG, optimizerD):
    checkpoint_G = torch.load(PATH_G)
    checkpoint_D = torch.load(PATH_D)
    G.load_state_dict(checkpoint_G['state_dict'])
    optimizerG.load_state_dict(checkpoint_G['optimizer'])
    D.load_state_dict(checkpoint_D['state_dict'])
    optimizerD.load_state_dict(checkpoint_D['optimizer'])
    prev_epochs = checkpoint_G['epoch']
    return G, D, optimizerG, optimizerD, prev_epochs

def xavier_init_weights(model):
    if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)

if __name__ == "__main__":
    train(resume_training=False)
