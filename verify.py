import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- Configurable flags ----
NORMALIZE = False
CROP_HEIGHT = 110   # Set 0 for no crop
USE_MEDIAN = False  # Set True for median, False for mean
deviation_threshold = 0.3   # <-- Set your deviation threshold here

# 1. Define the IJEPA Backbone
class IJEPABackbone(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256)):
        super().__init__()
        self.encoder = timm.create_model(
            vit_name, pretrained=False, num_classes=0, img_size=img_size
        )
    def forward(self, x):
        z = self.encoder.forward_features(x)  # (B, N+1, D)
        return z

# 2. Define the regression head
class SteeringRegressionHead(nn.Module):
    def __init__(self, encoder, out_dim=768):
        super().__init__()
        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, 100),
            nn.ELU(),
            nn.Linear(100, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        z = self.encoder.encoder.forward_features(x)
        z = z[:, 1:, :]  # remove CLS token
        z = z.permute(0, 2, 1)  # (B, D, N)
        pooled = self.avgpool(z).squeeze(-1)  # (B, D)
        out = self.regressor(pooled)
        return out.squeeze(-1)  # (B,)

# 3. Load Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the encoder once
encoder = IJEPABackbone(vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256)).to(device)

# Load 3 regression heads
straight_model = SteeringRegressionHead(encoder, out_dim=768).to(device)
left_model = SteeringRegressionHead(encoder, out_dim=768).to(device)
right_model = SteeringRegressionHead(encoder, out_dim=768).to(device)

# Define paths
ckpt_straight = '/scratch/vn2263/Predictive-Steering-with-I-JEPA/finetuned_straight_seq_model.pth'
ckpt_left = '/scratch/vn2263/Predictive-Steering-with-I-JEPA/finetuned_left_seq_model.pth'
ckpt_right = '/scratch/vn2263/Predictive-Steering-with-I-JEPA/finetuned_right_seq_model.pth'

# Load checkpoints
straight_model.load_state_dict(torch.load(ckpt_straight, map_location=device), strict=False)
left_model.load_state_dict(torch.load(ckpt_left, map_location=device), strict=False)
right_model.load_state_dict(torch.load(ckpt_right, map_location=device), strict=False)

straight_model.eval()
left_model.eval()
right_model.eval()

print("Loaded all 3 models.")

# 4. Image preprocessing
transform_list = [T.Resize((128, 256)), T.ToTensor()]
if NORMALIZE:
    transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
torch_transform = T.Compose(transform_list)

# 5. Data loading
img_dir = '/vast/vn2263/carla-dataset/Data/Images'
label_path = '/vast/vn2263/carla-dataset/Data/SteerValues/steer_values.txt'

img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')))
with open(label_path) as f:
    labels = [float(line.strip()) for line in f]

assert len(img_paths) == len(labels), "Mismatch between images and labels!"

# --- Select only deviating indices ---
if USE_MEDIAN:
    normal_value = np.median(labels)
else:
    normal_value = np.mean(labels)

selected_indices = [i for i, gt in enumerate(labels) if abs(gt - normal_value) > deviation_threshold]

print(f"Found {len(selected_indices)} out of {len(labels)} data points deviating more than {deviation_threshold} from normal value ({normal_value:.3f})")

filtered_img_paths = [img_paths[i] for i in selected_indices]
filtered_labels = [labels[i] for i in selected_indices]

# 6. Inference loop
preds, gts = [], []

for img_path, gt in tqdm(zip(filtered_img_paths, filtered_labels), total=len(filtered_img_paths), desc="Inference (deviating only)"):
    img = Image.open(img_path).convert('RGB')
    if CROP_HEIGHT > 0:
        img = img.crop((0, CROP_HEIGHT, img.width, img.height))
    img = torch_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_straight = straight_model(img).cpu().item()
        pred_left = left_model(img).cpu().item()
        pred_right = right_model(img).cpu().item()

    # Ensemble selection strategy
    if abs(pred_straight) < 0.1:  # threshold near zero
        pred = pred_straight
    elif pred_left < 0 and abs(pred_left) > abs(pred_right):
        pred = pred_left
    else:
        pred = pred_right

    preds.append(pred)
    gts.append(gt)

# 7. Metrics & Save results
preds = np.array(preds)
gts = np.array(gts)

mae = np.mean(np.abs(preds - gts))
mse = np.mean((preds - gts) ** 2)
print(f"\nMAE: {mae:.4f}, MSE: {mse:.4f}")

df = pd.DataFrame({
    'image_path': filtered_img_paths,
    'ground_truth': gts,
    'prediction': preds
})
csv_path = 'gt_vs_pred_deviating_ensemble.csv'
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# 8. Visualization
plt.figure(figsize=(12, 6))
plt.plot(gts, label='Ground Truth')
plt.plot(preds, label='Prediction')
plt.legend()
plt.title('Steering Value: Ground Truth vs Prediction (Deviating Data Only)')
plt.xlabel('Image Index')
plt.ylabel('Steering Value')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(gts, preds, alpha=0.3)
plt.plot([min(gts), max(gts)], [min(gts), max(gts)], 'r--', label='Ideal')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.title('GT vs Prediction Scatter Plot (Deviating Data Only)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Show worst predictions ---
top_n = 5
worst_indices = np.argsort(np.abs(preds - gts))[-top_n:]
print("\nWorst predictions (by abs error):")
for idx in worst_indices:
    print(f"Image: {filtered_img_paths[idx]} | GT: {gts[idx]:.3f} | Pred: {preds[idx]:.3f} | AbsErr: {abs(gts[idx] - preds[idx]):.3f}")

    # Optional: save worst images
    # img = Image.open(filtered_img_paths[idx])
    # img.save(f"worst_pred_{idx}.png")
