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

# --- Model definition (updated with avgpool + 3-layer regressor) ---
class IJEPABackbone(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256)):
        super().__init__()
        self.encoder = timm.create_model(
            vit_name, pretrained=False, num_classes=0, img_size=img_size
        )

    def forward(self, x):
        z = self.encoder.forward_features(x)  # (B, N+1, D)
        return z

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
        z = self.encoder(x)  # (B, N+1, D)
        z = z[:, 1:, :]  # Remove CLS token
        z = z.permute(0, 2, 1)  # (B, D, N)
        pooled = self.avgpool(z).squeeze(-1)  # (B, D)
        out = self.regressor(pooled)
        return out.squeeze(-1)  # (B,)

# --- Add CARLA egg if needed ---
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("no carla egg file found")

import carla
from carla import ColorConverter as cc

# --- Global for camera frame ---
global current_frame
current_frame = None

def get_img(image):
    global current_frame
    image = np.array(image.raw_data)
    img = image.reshape((220, 220, 4))[:, :, :3]
    img = img[110:220, :, :]  # Crop (like offline preprocessing)
    current_frame = img.copy()

# --- Torch transforms ---
torch_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 256)),
    T.ToTensor(),
])

# --- Load model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = IJEPABackbone(vit_name='vit_base_patch16_224', out_dim=768, img_size=(128, 256))
model = SteeringRegressionHead(encoder, out_dim=768).to(device)

ckpt_path = '/scratch/vn2263/Predictive-Steering-with-I-JEPA/finetuned_ijepa_deviant2_epoch20.pth'
state_dict = torch.load(ckpt_path, map_location=device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()

# --- Setup CARLA client ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print("Connected to CARLA server.")

world = client.load_world('Town02')
map_ = world.get_map()

# Spawn vehicle
blueprint_vehicle = world.get_blueprint_library().filter('model3')[0]
spawn_points = map_.get_spawn_points()
vehicle = world.try_spawn_actor(blueprint_vehicle, random.choice(spawn_points))
print("Vehicle spawned.")

# Spawn camera
blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
blueprint.set_attribute('image_size_x', '220')
blueprint.set_attribute('image_size_y', '220')
blueprint.set_attribute('fov', '120')
camera_transform = carla.Transform(carla.Location(x=1.5, z=3))

camera = world.try_spawn_actor(blueprint, camera_transform, attach_to=vehicle)
if not camera:
    raise RuntimeError("Failed to spawn camera.")
print("Camera spawned.")

camera.listen(lambda data: get_img(data))

# --- Driving loop ---
while True:
    if current_frame is None:
        continue

    # Preprocess current frame
    img = torch_transform(current_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)
        steer_value = float(pred.cpu().numpy()[0])

    # Control logic
    control = carla.VehicleControl()
    vel = vehicle.get_velocity()
    vel = 3.6 * math.sqrt(vel.x * 2 + vel.y * 2 + vel.z ** 2)  # m/s to km/h

    print(f"Velocity: {vel:.2f} km/h")
    if vel < 12:
        control.throttle = 0.2
        control.brake = 0
    else:
        control.throttle = 0
        control.brake = 1

    control.steer = np.clip(steer_value, -1.0, 1.0)
    print(f"Steering command: {control.steer:.3f}")

    vehicle.apply_control(control)
    time.sleep(0.1)

    # Show current frame
    cv2.imshow("Current Camera View", current_frame)
    cv2.waitKey(10)
