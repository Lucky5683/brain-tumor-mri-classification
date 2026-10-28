
from pathlib import Path
import tensorflow as tf

checkpoint = Path(
    r"experiments\spatial_fusion\meningioma_weighted_50ep_seed42\best_model.keras"
)

if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint.resolve()}")

model = tf.keras.models.load_model(checkpoint, compile=False)

print("\n===== SELECTED MODEL =====")
print("Name:", model.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Total parameters:", f"{model.count_params():,}")

print("\n===== TOP-LEVEL LAYERS =====")
for index, layer in enumerate(model.layers):
    output_shape = getattr(layer, "output_shape", "not directly available")
    print(f"{index:3} | {layer.name:35} | {layer.__class__.__name__:25} | {output_shape}")

print("\n===== NESTED MODEL LAYERS =====")
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        print(f"\nNested model: {layer.name}")
        for sublayer in layer.layers:
            if any(term in sublayer.name.lower() for term in
                   ["efficientnet", "block3b", "top_activation", "fusion"]):
                print(sublayer.name, "|", sublayer.__class__.__name__)

print("\n===== CLASS OUTPUT =====")
print("Expected classes: glioma, meningioma, notumor, pituitary")
