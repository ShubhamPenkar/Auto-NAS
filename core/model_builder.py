"""
model_builder.py
----------------
Builds and trains Keras neural networks from genomes.
Classification only.

Architecture per hidden layer:
    Dense(units, activation, L2) → BatchNormalization → Dropout(rate)

Improvements:
  - Evolved optimizer (adam / rmsprop / adamw) used per genome
  - L2 regularization on all Dense layers (reduces overfitting)
  - Early stopping patience increased 4 → 7 (fairer evaluation)
  - Complexity penalty removed from fitness (was biasing against NAS)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers


def _get_optimizer(genome: dict):
    """Return the Keras optimizer specified by the genome."""
    name = genome.get('optimizer', 'adam')
    lr   = 0.001
    if name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=lr)
    elif name == 'adamw':
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    else:  # default: adam
        return keras.optimizers.Adam(learning_rate=lr)


def build_model(genome: dict, input_dim: int, num_classes: int) -> keras.Model:
    """
    Dynamically build a Keras classifier from a genome dict.

    Args:
        genome:      dict with 'layers', 'activation', 'dropout', 'optimizer'
        input_dim:   number of input features
        num_classes: number of target classes
    Returns:
        Compiled Keras model
    """
    tf.get_logger().setLevel('ERROR')

    layers     = genome['layers']
    activation = genome['activation']
    dropout    = genome['dropout']

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    for units in layers:
        model.add(keras.layers.Dense(
            units, activation=activation,
            kernel_regularizer=regularizers.l2(1e-4)
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))

    optimizer = _get_optimizer(genome)

    if num_classes == 2:
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def compute_fitness(
    genome:      dict,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    problem_type: str,          # kept for API compatibility, always classification
    num_classes:  int   = None,
    epochs:       int   = 20,
    complexity_penalty: float = 0.0   # kept for API compatibility, not used
) -> tuple:
    """
    Train a model from a genome and return its fitness (val_accuracy).

    Returns:
        (fitness_score, num_parameters, history)
    """
    input_dim = X_train.shape[1]
    model     = build_model(genome, input_dim, num_classes)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,                    # increased from 4 — gives slow starters a fair chance
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    num_params = model.count_params()

    # Pure val_accuracy — no complexity penalty (baselines have none either)
    fitness = max(history.history['val_accuracy'])

    keras.backend.clear_session()
    del model

    return round(fitness, 5), num_params, history


def build_final_model(
    genome:       dict,
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    problem_type: str,          # kept for API compatibility
    num_classes:  int = None,
    epochs:       int = 100
) -> tuple:
    """
    Build and fully train the best model with early stopping + LR scheduling.

    Returns:
        (trained Keras model, history object)
    """
    input_dim = X_train.shape[1]
    model     = build_model(genome, input_dim, num_classes)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop, lr_scheduler]
    )

    return model, history
