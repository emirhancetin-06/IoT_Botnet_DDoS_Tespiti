"""Keras modelleri: önerilen hafif hibrit CNN-LSTM ve kıyaslama modelleri.

Tüm kurucu fonksiyonlar aynı arayüze sahiptir::

    model = Lightweight_CNN_LSTM(input_shape=(k, 1), num_classes=4)

ve derlenmiş (Adam + sparse_categorical_crossentropy) bir ``keras.Model`` döndürür.
Girdi: k adet seçilmiş akış özniteliği, tek kanallı bir dizi olarak (k, 1).
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from tensorflow import keras

layers = keras.layers

LEARNING_RATE = 1e-3


def _compile(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def Lightweight_CNN_LSTM(input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
    """Önerilen model: [Conv1D-BN-MaxPool-Dropout] x2 -> LSTM -> Dense -> Softmax (~20 bin parametre)."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return _compile(keras.Model(inputs, outputs, name="Lightweight_CNN_LSTM"))


def Standalone_CNN(input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
    """Yalnızca evrişimli kıyaslama modeli."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return _compile(keras.Model(inputs, outputs, name="Standalone_CNN"))


def Standalone_LSTM(input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
    """Yalnızca LSTM'li kıyaslama modeli."""
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(64)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return _compile(keras.Model(inputs, outputs, name="Standalone_LSTM"))


def Standalone_GRU(input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
    """Yalnızca GRU'lu kıyaslama modeli."""
    inputs = keras.Input(shape=input_shape)
    x = layers.GRU(64)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return _compile(keras.Model(inputs, outputs, name="Standalone_GRU"))


MODEL_BUILDERS: Dict[str, Callable[[Tuple[int, int], int], keras.Model]] = {
    "CNN": Standalone_CNN,
    "LSTM": Standalone_LSTM,
    "GRU": Standalone_GRU,
    "Lightweight_CNN_LSTM": Lightweight_CNN_LSTM,
}
