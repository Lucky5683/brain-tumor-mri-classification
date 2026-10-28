
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_multiscale_fusion_model(
    checkpoint_path="models/efficientnet_b0_finetuned.keras",
    num_classes=4,
    dropout_rate=0.35,
):
    original = tf.keras.models.load_model(
        checkpoint_path,
        compile=False,
    )

    backbone = original.get_layer("efficientnetb0")

    # Connect both feature maps to the same backbone input.
    feature_extractor = Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("block6d_add").output,
            backbone.get_layer("top_activation").output,
        ],
        name="multiscale_feature_extractor",
    )

    image_input = layers.Input(
        shape=(224, 224, 3),
        name="fusion_image_input",
    )

    intermediate_features, final_features = feature_extractor(
        image_input
    )

    intermediate_vector = layers.GlobalAveragePooling2D(
        name="intermediate_pool"
    )(intermediate_features)

    final_vector = layers.GlobalAveragePooling2D(
        name="final_pool"
    )(final_features)

    fused = layers.Concatenate(name="multiscale_fusion")(
        [intermediate_vector, final_vector]
    )

    x = layers.Dropout(dropout_rate, name="fusion_dropout")(fused)
    x = layers.Dense(256, activation="relu", name="fusion_dense")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="fusion_predictions",
    )(x)

    return Model(
        inputs=image_input,
        outputs=outputs,
        name="EfficientNetB0_MultiScaleFusion",
    )


if __name__ == "__main__":
    model = build_multiscale_fusion_model()
    print("Model:", model.name)
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)
    print("Parameters:", model.count_params())
    model.summary()
