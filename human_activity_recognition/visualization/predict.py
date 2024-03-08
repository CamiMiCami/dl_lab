"""
This script was developed by Shuaike Liu as a script for
Predicting the class, overwrite with 0, and save as txt file:
y_pred.txt

Bi_GRU: from Bao in models.py
compute_accuracy: calculate the accuracy of prediction and truth
predict: make prediction using Bi_GRU
"""
import numpy as np
import tensorflow as tf
from human_activity_recognition.Visual import prepare_visu_data


def Bi_GRU(model_archi):
    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    inputs = tf.keras.Input(shape=input_shape)

    # GRU layer + 2 dense layers
    out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=rec_unit, activation='relu', return_sequences=True))(
        inputs)
    out = tf.keras.layers.Dense(units=dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='GRU_class')


def compute_accuracy(y_true, y_pred):
    # Ensure both arrays have the same length
    assert len(y_true) == len(y_pred), "Arrays must have the same length"

    # Count the number of matching elements
    correct_predictions = np.sum(y_true == y_pred)

    # Compute accuracy as the percentage of correct predictions
    accuracy = (correct_predictions / len(y_true)) * 100

    return accuracy


def predict():
    sensor_data_raw = np.loadtxt(r"user22_data")
    y_true = np.loadtxt(r"user22_labels")

    sensor_data = tf.data.Dataset.from_tensor_slices(sensor_data_raw)
    sensor_data = sensor_data.window(size=250, shift=250, drop_remainder=True)
    sensor_data = sensor_data.flat_map(lambda window: window.batch(250))
    sensor_data = sensor_data.batch(50)
    sensor_data = sensor_data.prefetch(tf.data.experimental.AUTOTUNE)

    input_shape = (250, 6)
    rec_unit = 64
    dense_units = 32
    dropout_rate = 0.25
    num_classes = 12
    model_archi = input_shape, rec_unit, dense_units, dropout_rate, num_classes
    model = Bi_GRU(model_archi)
    checkpoint = "ckpt-469"
    tf.train.Checkpoint(model=model).restore(checkpoint)

    flat_y_pred = []

    for window in sensor_data:
        y_pred = model.predict(window)
        y_pred = tf.reshape(y_pred, (-1, 12))
        y_pred = tf.argmax(y_pred, axis=1)
        flat_y_pred.append(y_pred)

    flat_y_pred = tf.concat(flat_y_pred, axis=0).numpy()
    flat_y_pred = flat_y_pred+1
    y_true = y_true[:len(flat_y_pred)]
    zero_indices = np.where(y_true == 0)[0]

    for index in zero_indices:
        flat_y_pred[index] = 0
    np.savetxt("y_pred", flat_y_pred)

    return print(compute_accuracy(y_true, flat_y_pred))


predict()
