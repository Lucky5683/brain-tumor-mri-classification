
from pathlib import Path
import tensorflow as tf
import src.models.compat

checkpoint = Path("models/multiscale_fusion_grouped_best.keras")

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint.resolve())

model = tf.keras.models.load_model(checkpoint, compile=False)

print("Model:", model.name)
print("Input:", model.input_shape)
print("Output:", model.output_shape)

for layer in model.layers:
    print(f"{layer.name:40s} {type(layer).__name__}")

extractor = model.get_layer("multiscale_feature_extractor")
print("\nFeature extractor outputs:")
for output in extractor.outputs:
    print(output.shape)

print("\nCheckpoint parameters:", model.count_params())
