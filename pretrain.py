import os
import glob
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

# ===== SETUP LOGGER =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== 1. DATASET & DATALOADER =====

class CarlaImageDataset(Dataset):
    def __init__(self, img_dir, image_size=(128, 256)):
        self.img_paths = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img

# ===== 2. PATCH MASKING FUNCTION =====

def mask_random_patches(images, patch_size=16, mask_ratio=0.5):
    # images: (B, C, H, W)
    B, C, H, W = images.shape
    num_patches_H, num_patches_W = H // patch_size, W // patch_size
    num_total_patches = num_patches_H * num_patches_W
    num_mask = int(mask_ratio * num_total_patches)
    masks = torch.zeros((B, num_patches_H, num_patches_W), dtype=torch.bool)
    for i in range(B):
        mask_idx = torch.randperm(num_total_patches)[:num_mask]
        masks[i].view(-1)[mask_idx] = 1
    return masks  # (B, num_patches_H, num_patches_W)

# ===== 3. MODEL (I-JEPA BACKBONE + PREDICTOR) =====

import timm

class IJEPABackbone(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256)):
        super().__init__()
        self.encoder = timm.create_model(
            vit_name, pretrained=False, num_classes=0, img_size=img_size
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        z = self.encoder.forward_features(x)  # (B, N+1, D)
        z = z[:, 1:, :]                       # Remove CLS token
        z_pred = self.predictor(z)
        return z, z_pred

def update_ema(ema_model, model, beta=0.996):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(beta).add_(param.data, alpha=1 - beta)

# ===== 4. LOSSES =====

def cosine_loss(pred, target, mask):
    # pred, target: (B, N, D)
    # mask: (B, N_patches_H, N_patches_W)
    B, N, D = pred.shape
    mask = mask.flatten(1)  # (B, N)
    # Flatten batch and patch dims for masked elements
    pred_masked = pred[mask]
    target_masked = target[mask]
    # cosine similarity, want to maximize, so use -cosine for loss
    if pred_masked.shape[0] == 0:  # sometimes, especially if mask_ratio is low, may be empty!
        return torch.tensor(0.0, device=pred.device)
    cos_sim = F.cosine_similarity(pred_masked, target_masked, dim=-1)
    return 1 - cos_sim.mean()

def variance_loss(z):
    # z: (B, N, D) -> flatten over batch and patch dims
    z = z.reshape(-1, z.shape[-1])
    std = z.std(dim=0) + 1e-4
    return torch.mean(F.relu(1 - std))

# ===== 5. MAIN TRAINING LOOP =====

def main():
    # --------- Hyperparameters ---------
    image_dir = '/vast/vn2263/carla-dataset/Data/Images'
    image_size = (128, 256)
    patch_size = 16
    mask_ratio = 0.5
    batch_size = 128
    num_workers = 8
    epochs = 100
    lambda_var = 25.0
    ema_decay = 0.996
    vit_name ='vit_base_patch16_224'
    out_dim = 768
    lr = 1e-4

    # --------- Dataset & Dataloader ---------
    dataset = CarlaImageDataset(image_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # --------- Model & EMA Model ---------
    ijepa = IJEPABackbone(vit_name, out_dim=out_dim).cuda()
    ema_ijepa = copy.deepcopy(ijepa).cuda().eval()
    optimizer = torch.optim.AdamW(ijepa.parameters(), lr=lr)

    # --------- Training ---------
    for epoch in range(epochs):
        ijepa.train()
        total_loss, total_cos, total_var = 0, 0, 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for images in pbar:
            images = images.cuda()
            masks = mask_random_patches(images, patch_size=patch_size, mask_ratio=mask_ratio).cuda()
            
            # Online encoder and predictor
            z_c, z_pred = ijepa(images)   # (B, N, D)
            # EMA target encoder
            with torch.no_grad():
                z_t, _ = ema_ijepa(images)

            # Cosine loss on masked patches
            cos_loss = cosine_loss(z_pred, z_t, masks)
            # Variance regularization
            var_loss = variance_loss(z_c) + variance_loss(z_pred)
            # Total loss
            loss = cos_loss + lambda_var * var_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            update_ema(ema_ijepa, ijepa, beta=ema_decay)

            total_loss += loss.item()
            total_cos += cos_loss.item()
            total_var += var_loss.item()
            pbar.set_postfix({'Loss': loss.item(), 'Cos': cos_loss.item(), 'Var': var_loss.item()})
        logger.info(
            f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Cos={total_cos/len(dataloader):.4f}, Var={total_var/len(dataloader):.4f}"
        )

        # ===== Save model every 5 epochs =====
        if (epoch + 1) % 5 == 0:
            save_path = f'ijepa_carla_pretrain_epoch{epoch+1}.pth'
            torch.save(ijepa.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    # ===== Final save after all epochs =====
    torch.save(ijepa.state_dict(), 'ijepa_carla_pretrain_final.pth')
    logger.info("Final model saved to ijepa_carla_pretrain_final.pth")

if __name__ == '__main__':
    main()
