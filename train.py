import tensorflow.compat.v2 as tf
import cv2
import glob
import numpy as np
from tqdm import tqdm
from AD_L_JEPA import ADLJEPA

# Paths
path = "/vast/vn2263/carla-dataset/Data/SteerValues/steer_values.txt"
img_folder = "/vast/vn2263/carla-dataset/Data/Images/*.png"

# ----------------
# Load images with progress bar
# ----------------
img_paths = sorted(glob.glob(img_folder))
imgs = []
for img in tqdm(img_paths, desc="Loading images"):
    image = cv2.imread(img)[110:220, :, :1]  # Crop and keep single channel
    imgs.append(image)

# ----------------
# Load labels with progress bar
# ----------------
labels = []
# First, count number of lines for tqdm
with open(path, "r") as f:
    n_lines = sum(1 for _ in f)
with open(path, "r") as file1:
    for val in tqdm(file1, total=n_lines, desc="Loading labels"):
        val = float(val.strip()) * 100  # Scale as before
        labels.append(tf.cast(val, tf.float32))

# -------------------------------
# Ensure arrays are same length
# -------------------------------
n = min(len(imgs), len(labels))
imgs_np = np.asarray(imgs[:n], dtype=np.float32)
labels_np = np.asarray(labels[:n], dtype=np.float32)

# Normalize images to [0, 1]
imgs_np = imgs_np / 255.0

# -----------------------------------------
# Build AD-L-JEPA-based Regression Model
# -----------------------------------------
bev_shape = (110, 220, 1)
embed_dim = 128
encoder_depth = 4
predictor_depth = 3

jepa_core = ADLJEPA(
    bev_shape=bev_shape, 
    embed_dim=embed_dim, 
    encoder_depth=encoder_depth, 
    predictor_depth=predictor_depth
)

input_layer = tf.keras.Input(shape=bev_shape)
context_embeds, pred_embeds = jepa_core(input_layer)
gap = tf.keras.layers.GlobalAveragePooling2D()(pred_embeds)
output = tf.keras.layers.Dense(1)(gap)

regression_model = tf.keras.Model(inputs=input_layer, outputs=output)

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
regression_model.compile(
    loss="mean_squared_error", 
    optimizer=opt, 
    metrics=["mae"]
)

# -----------------------------------------
# Keras already gives a training progress bar, but if you want tqdm for epochs:
# -----------------------------------------
EPOCHS = 10
BATCH_SIZE = 256

for epoch in tqdm(range(EPOCHS), desc="Training epochs"):
    regression_model.fit(
        imgs_np, labels_np,
        epochs=epoch+1,
        initial_epoch=epoch,
        batch_size=BATCH_SIZE,
        verbose=1  # Keras will still show its batch progress bar
    )

regression_model.save("steer_model")
