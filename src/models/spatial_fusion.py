
import tensorflow as tf
from tensorflow.keras import Model, layers


def build_spatial_fusion_model(
    checkpoint_path="models/efficientnet_b0_finetuned.keras",
    num_classes=4,
    dropout_rate=0.35,
):
    original = tf.keras.models.load_model(
        checkpoint_path,
        compile=False,
    )

    backbone = original.get_layer("efficientnetb0")

    # Preserve spatial information from an earlier EfficientNet block.
    extractor = Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("block3b_add").output,
            backbone.get_layer("top_activation").output,
        ],
        name="spatial_feature_extractor",
    )

    image_input = layers.Input(
        shape=(224, 224, 3),
        name="spatial_image_input",
    )

    early_features, deep_features = extractor(image_input)

    # Project early features: (28, 28, 40) -> (28, 28, 64).
    early = layers.Conv2D(
        64, 1, padding="same", use_bias=False,
        name="early_projection",
    )(early_features)
    early = layers.BatchNormalization(name="early_bn")(early)
    early = layers.Activation("swish", name="early_swish")(early)

    # Reduce spatial resolution: (28, 28, 64) -> (7, 7, 64).
    early = layers.AveragePooling2D(
        pool_size=4,
        strides=4,
        name="early_downsample",
    )(early)

    # Project deep features to a compact representation.
    deep = layers.Conv2D(
        128, 1, padding="same", use_bias=False,
        name="deep_projection",
    )(deep_features)
    deep = layers.BatchNormalization(name="deep_bn")(deep)
    deep = layers.Activation("swish", name="deep_swish")(deep)

    # Fuse spatial features before pooling.
    fused = layers.Concatenate(name="spatial_fusion")([early, deep])
    fused = layers.Conv2D(
        128, 3, padding="same", use_bias=False,
        name="fusion_conv",
    )(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    fused = layers.Activation("swish", name="fusion_swish")(fused)

    # Classification head.
    x = layers.GlobalAveragePooling2D(name="fusion_pool")(fused)
    x = layers.Dropout(dropout_rate, name="fusion_dropout")(x)
    x = layers.Dense(256, activation="relu", name="fusion_dense")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="spatial_predictions",
    )(x)

    return Model(
        inputs=image_input,
        outputs=outputs,
        name="EfficientNetB0_SpatialFusion",
    )


if __name__ == "__main__":
    model = build_spatial_fusion_model()
    print("Model:", model.name)
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)
    print("Parameters:", f"{model.count_params():,}")
    model.summary()
