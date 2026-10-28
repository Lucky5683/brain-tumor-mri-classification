
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401

checkpoint = ROOT / "models" / "multiscale_fusion_grouped_best.keras"

model = tf.keras.models.load_model(checkpoint, compile=False)
extractor = model.get_layer("multiscale_feature_extractor")

extractor.trainable = True

for layer in extractor.layers:
    layer.trainable = False

eligible = [
    layer for layer in extractor.layers
    if not isinstance(layer, (
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.InputLayer,
    ))
]

selected = eligible[-20:]

for layer in selected:
    layer.trainable = True

for layer in extractor.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

for layer in model.layers:
    if layer is not extractor:
        layer.trainable = True

trainable_params = sum(
    int(np.prod(w.shape)) for w in model.trainable_weights
)

bn_trainable = [
    layer.name
    for layer in extractor.layers
    if isinstance(layer, tf.keras.layers.BatchNormalization)
    and layer.trainable
]

print("Checkpoint loaded:", checkpoint.name, flush=True)
print("Feature extractor:", extractor.name, flush=True)
print("Selected layers:", [layer.name for layer in selected], flush=True)
print("Trainable parameters:", f"{trainable_params:,}", flush=True)
print("Trainable BatchNormalization layers:", bn_trainable, flush=True)

assert trainable_params > 0, "No trainable parameters!"
assert not bn_trainable, "BatchNormalization must remain frozen!"

print("PASS: Fine-tuning setup is valid.", flush=True)
