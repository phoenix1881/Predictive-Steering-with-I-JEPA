import glob
import os
import sys
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms as T
import time
import math
import timm
import carla
from carla import ColorConverter as cc
from PIL import Image

# --- Model backbone ---
class IJEPABackbone(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256)):
        super().__init__()
        self.encoder = timm.create_model(
            vit_name, pretrained=False, num_classes=0, img_size=img_size
        )
    def forward(self, x):
        z = self.encoder.forward_features(x)  # (B, N+1, D)
        return z

# --- Full model with regression head ---
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
        z = self.encoder.encoder.forward_features(x)  # (B, N+1, D)
        z = z[:, 1:, :]  # Remove CLS token
        z = z.permute(0, 2, 1)  # (B, D, N)
        pooled = self.avgpool(z).squeeze(-1)  # (B, D)
        out = self.regressor(pooled)
        return out.squeeze(-1)  # (B,)

# --- Globals ---
global current_frame
current_frame = None

# --- Camera callback ---
def get_img(image):
    global current_frame
    img = np.array(image.raw_data)
    img = img.reshape((220, 220, 4))[:, :, :3]  # Take RGB only
    img = img[110:220, :, :]  # Crop lower half (like your offline preprocessing)
    current_frame = img

# --- Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = IJEPABackbone(vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256))
model = SteeringRegressionHead(encoder, out_dim=768).to(device)

ckpt_path = '/scratch/vn2263/Predictive-Steering-with-I-JEPA/finetuned_ijepa_deviant2_epoch20.pth'
state_dict = torch.load(ckpt_path, map_location=device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
model.eval()

# --- Transformations ---
torch_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 256)),
    T.ToTensor(),
])

# --- Connect to CARLA ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print("Connected to CARLA.")

world = client.load_world('Town02')
map_ = world.get_map()

# --- Spawn Vehicle ---
vehicle_bp = world.get_blueprint_library().filter('model3')[0]
spawn_points = map_.get_spawn_points()
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
if vehicle is None:
    raise RuntimeError("Failed to spawn vehicle.")
print("Vehicle spawned.")

# --- Attach Camera ---
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '220')
camera_bp.set_attribute('image_size_y', '220')
camera_bp.set_attribute('fov', '120')

camera_transform = carla.Transform(carla.Location(x=1.5, z=3))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(lambda image: get_img(image))
print("Camera spawned and listening.")

# --- Driving loop ---
try:
    while True:
        if current_frame is None:
            continue

        # Preprocess the frame
        img = torch_transform(current_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img)
            steer_value = float(pred.cpu().numpy()[0])

        # Prepare control
        control = carla.VehicleControl()
        vel = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # m/s to km/h

        # Simple throttle-brake logic
        if speed < 12:
            control.throttle = 0.3
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = 0.2

        control.steer = np.clip(steer_value, -1.0, 1.0)
        vehicle.apply_control(control)

        # (Optional) visualize
        cv2.imshow("Current Frame", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

finally:
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
    print("Shutdown complete.")
