import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from pretrain import IJEPABackbone

# === 1. Mixed Deviant + Normal Dataset (70% deviant, 30% normal) ===

class CarlaSteeringMixedDataset(Dataset):
    def __init__(self, img_dir, label_path, image_size=(128, 256), threshold=0.1, deviant_frac=0.7):
        img_list = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')))
        with open(label_path, "r") as f:
            labels = [float(x.strip()) for x in f.readlines()]
        n = min(len(img_list), len(labels))
        img_list = img_list[:n]
        labels = labels[:n]
        mean_label = np.mean(labels)

        # Split indices into deviant and normal
        deviant_indices = [i for i, l in enumerate(labels) if abs(l - mean_label) > threshold]
        normal_indices = [i for i, l in enumerate(labels) if abs(l - mean_label) <= threshold]

        num_deviant = int(deviant_frac * len(deviant_indices))
        num_normal = int((1 - deviant_frac) * len(deviant_indices))  # Keep normal smaller, proportional to deviants

        selected_deviant = np.random.choice(deviant_indices, size=num_deviant, replace=False)
        selected_normal = np.random.choice(normal_indices, size=num_normal, replace=False)

        final_indices = np.concatenate([selected_deviant, selected_normal])
        np.random.shuffle(final_indices)

        self.img_list = [img_list[i] for i in final_indices]
        self.labels = [labels[i] for i in final_indices]

        # Diagnostics
        labels_arr = np.array(self.labels)
        print(f"Selected {len(self.labels)} samples: {np.sum(labels_arr > 0)} positive, {np.sum(labels_arr < 0)} negative")
        print(f"Mean label: {np.mean(labels_arr):.4f}, Std label: {np.std(labels_arr):.4f}")

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])
        self.mean = float(np.mean(self.labels))
        self.std = float(np.std(self.labels)) + 1e-6
        self.labels_norm = [(lbl - self.mean) / self.std for lbl in self.labels]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels_norm[idx], dtype=torch.float32)
        orig_label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label, orig_label

# === 2. Model ===

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
        z = z[:, 1:, :]
        z = z.permute(0, 2, 1)
        pooled = self.avgpool(z).squeeze(-1)
        out = self.regressor(pooled)
        return out.squeeze(-1)

# === 3. Loss ===

def combined_loss(pred, target, alpha=1.0, beta=0.1):
    mse = (pred - target) ** 2
    cube = torch.abs(pred - target) ** 3
    loss = alpha * mse + beta * cube
    return loss.mean()

# === 4. Train and Evaluate ===

def main():
    img_dir = '/vast/vn2263/carla-dataset/Data/Images'
    label_path = '/vast/vn2263/carla-dataset/Data/SteerValues/steer_values.txt'
    image_size = (128, 256)
    batch_size = 64
    epochs = 20
    lr = 1e-4
    weight_path = 'ijepa_carla_pretrain_final1.pth'
    out_dim = 768
    threshold = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Mixed dataset: 70% deviant + 30% normal
    dataset = CarlaSteeringMixedDataset(img_dir, label_path, image_size=image_size, threshold=threshold, deviant_frac=0.7)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Training on {len(dataset)} samples: 70% turning cases + 30% normal cases")

    # Load model
    ijepa = IJEPABackbone(vit_name='vit_base_patch16_224', out_dim=out_dim, img_size=image_size)
    ijepa.load_state_dict(torch.load(weight_path, map_location=device))

    # Freeze encoder
    for param in ijepa.parameters():
        param.requires_grad = False
    # (Optional) unfreeze last 2 blocks
    # for name, param in ijepa.encoder.named_parameters():
    #     if "blocks.10" in name or "blocks.11" in name:
    #         param.requires_grad = True

    model = SteeringRegressionHead(ijepa, out_dim=out_dim).to(device)
    for param in model.regressor.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels, orig_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"finetuned_ijepa_mixed_epoch{epoch+1}.pth")
        print(f"Model saved to finetuned_ijepa_mixed_epoch{epoch+1}.pth")

    # Evaluation
    print("\nEvaluating on mixed samples:")
    model.eval()
    all_preds, all_gts = [], []
    mean, std = dataset.mean, dataset.std
    neg_examples = []
    with torch.no_grad():
        for imgs, labels, orig_labels in tqdm(dataloader):
            imgs = imgs.to(device)
            preds = model(imgs)
            preds_denorm = preds.cpu().numpy() * std + mean
            labels_denorm = orig_labels.numpy()
            all_preds.extend(preds_denorm.tolist())
            all_gts.extend(labels_denorm.tolist())
            for gt, pred in zip(labels_denorm, preds_denorm):
                if gt < 0:
                    neg_examples.append((gt, pred))

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(all_gts, all_preds)
    mse = mean_squared_error(all_gts, all_preds)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

    # Print some negative examples
    print("\nSample negative GTs and their predictions:")
    for gt, pred in neg_examples[:20]:
        print(f"GT: {gt:.3f} | Pred: {pred:.3f}")

    # Save predictions
    import pandas as pd
    df = pd.DataFrame({'gt': all_gts, 'pred': all_preds})
    df.to_csv("mixed_eval_results.csv", index=False)
    print("Saved results to mixed_eval_results.csv")

if __name__ == "__main__":
    main()
