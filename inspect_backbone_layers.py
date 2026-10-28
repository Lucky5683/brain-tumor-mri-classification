print("STEP 1: Script started", flush=True)

import tensorflow as tf
import src.models.compat

print("STEP 2: Loading checkpoint", flush=True)

model = tf.keras.models.load_model(
    "models/efficientnet_b0_finetuned.keras",
    compile=False,
)

print("STEP 3: Checkpoint loaded", flush=True)

backbone = model.get_layer("efficientnetb0")

print("Backbone:", backbone.name, flush=True)

for name in [
    "block3b_add",
    "block4c_add",
    "block6d_add",
    "top_activation",
]:
    try:
        layer = backbone.get_layer(name)
        print(f"{name}: {layer.output.shape}", flush=True)
    except ValueError:
        print(f"{name}: NOT FOUND", flush=True)

print("STEP 4: Inspection complete", flush=True)
