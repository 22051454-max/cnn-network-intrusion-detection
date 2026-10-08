"""1-D CNN over the 69-feature flow vector."""


def build_model(n_features, n_classes):
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(32, 3, padding="same", activation="relu", name="conv1"),
        layers.MaxPooling1D(2, name="pool1"),
        layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2"),
        layers.MaxPooling1D(2, name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax", name="out"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_numpy_weights(model, path):
    """Save weights as .npz so the demo app can run inference with numpy only (no TensorFlow)."""
    import numpy as np
    arrays = {}
    for name in ("conv1", "conv2", "dense1", "out"):
        w, b = model.get_layer(name).get_weights()
        arrays[f"{name}_w"], arrays[f"{name}_b"] = w, b
    np.savez_compressed(path, **arrays)
