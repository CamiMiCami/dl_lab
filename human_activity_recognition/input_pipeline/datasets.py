'''
This script was developed by Shuaike Liu, including 4 functions.

Function load(file_raw, save_to_txt=False) is used to generate datasets. 
It reads all the txt file as raw data in dataset folder, processes them and returns train_ds, val_ds, test_ds
Parameter file_raw is the DIR of the dataset folder which is configured in gin config file.
Parameter save_to_txt determined if a txt document is saved or not. 
If True, a txt document will be saved locally after data processing, which can save processing time for next run.

Function load_dataset() is used to read locally txt files which are saved by function load(save_to_txt=True) in previous run
and returns train_ds, val_ds, test_ds

Function dataset_for_oversample() is used to pick samples with class 7,8,9,10,11,12 for oversampling. 
It is called in function prepare() for oversampling

Function prepare(train_ds, val_ds, test_ds, batch_size=300 , repeats=0,std=0) is used to prepare datasets after load() or load_dataset()
It calls dataset_for_oversample() for oversampling and aug() for adding noise to samples
Batch size, repeat times for oversampling class 7~12 (repeats) and strength of added noise to samples (std) can be defined here. 
'''

import tensorflow as tf
import numpy as np
import os
import re
import logging
from input_pipeline.preprocess import z_score, aug
import gin

def save_txt(array, file_path):
    np.savetxt(file_path, array)

@gin.configurable
def load(file_raw, save_to_txt=False):
    logging.info("Preparing dataset...")
    file_label = file_raw+"/labels.txt"

    acc_ds = np.empty((0, 6))
    gyro_ds = np.empty((0, 3))
    labels = np.loadtxt(file_label)
    exp = []
    user = []

    """Get raw sensor data"""
    acc_pattern = r"^acc_exp(\d+)_user(\d+).txt$"
    gyro_pattern = r"^gyro_exp(\d+)_user(\d+).txt$"

    for filename in sorted(os.listdir(file_raw)):
        acc_match = re.match(acc_pattern, filename)
        gyro_match = re.match(gyro_pattern, filename)

        if acc_match:
            data = np.loadtxt(os.path.join(file_raw, filename))
            row_indices = np.arange(1, data.shape[0] + 1)[:, np.newaxis]
            data = np.concatenate((row_indices, data), axis=1)
            exp_value = int(acc_match.group(1))
            user_value = int(acc_match.group(2))
            exp = np.append(exp, exp_value)
            user = np.append(user, user_value)
            data = data[250:-250, :]  # remove 5 seconds

            # Add two columns for exp and user
            exp_column = np.full((data.shape[0], 1), exp_value)
            user_column = np.full((data.shape[0], 1), user_value)
            data_with_info = np.concatenate((exp_column, user_column, data), axis=1)
            acc_ds = np.concatenate((acc_ds, data_with_info), axis=0)

        elif bool(re.match(gyro_pattern, filename)):
            data = np.loadtxt(os.path.join(file_raw, filename))
            data = data[250:-250, :]  # remove 5 seconds
            gyro_ds = np.concatenate((gyro_ds, data), axis=0)

    sensor_data = np.concatenate((acc_ds, gyro_ds), axis=1) # All sensor data are here
    print(np.shape(sensor_data))

    """Get raw label data in order"""
    labels_ds = []
    # Iterate over the rows of the first array
    for i, (exp1, user1, time1, ax1, ay1, az1, gx, gy, gz) in enumerate(sensor_data):
        label_found = False

        # Iterate over the rows of the second array
        for exp2, user2, label, time_begin, time_end in labels:
            # Check conditions: exp, user, and time range
            if exp1 == exp2 and user1 == user2 and time_begin <= time1 <= time_end:
                labels_ds.append(label)
                label_found = True
                break

        # If no matching label is found, append 0
        if not label_found:
            labels_ds.append(0)

    # Convert the list to a numpy array
    labels_ds = np.array(labels_ds)
    labels_ds = np.transpose(labels_ds)
    save_txt(sensor_data, "sensor_whole")
    print(labels_ds)

    """Split dataset into train, val, test"""
    train_split_column_index = 1
    train_split_end_value = 22  # Change this to the first value you are looking for

    # Find the index where the specified column value first appears
    train_split_index = np.where(sensor_data[:, train_split_column_index] == train_split_end_value)[0]

    # Split the array into two parts based on the first appearance
    train_sensor_data = sensor_data[:train_split_index[0], :]
    remaining_data = sensor_data[train_split_index[0]:, :]

    # Specify the column index for the second split
    second_split_column_index = 1
    second_split_target_value = 28  # Change this to the second value you are looking for

    # Find the index where the second specified column value first appears
    second_split_index = np.where(remaining_data[:, second_split_column_index] == second_split_target_value)[0]

    # Split the remaining data into two parts based on the second appearance
    test_sensor_data = remaining_data[:second_split_index[0], :]
    val_sensor_data = remaining_data[second_split_index[0]:, :]

    """Split labels_ds for train, test, val"""
    first_split_index = len(train_sensor_data)
    second_split_index = len(test_sensor_data)

    # Split the fourth array into three parts based on the previously obtained indices
    train_labels = labels_ds[:first_split_index]
    test_labels = labels_ds[first_split_index:first_split_index+second_split_index]
    val_labels = labels_ds[first_split_index+second_split_index:]

    """Remove exp, user and time column from *_sensor_data and label"""
    train_sensor_data = train_sensor_data[:, 3:]
    test_sensor_data = test_sensor_data[:, 3:]
    val_sensor_data = val_sensor_data[:, 3:]

    """Z-Score Normalization"""
    train_sensor_data = z_score(train_sensor_data)
    test_sensor_data = z_score(test_sensor_data)
    val_sensor_data = z_score(val_sensor_data)

    """Save files"""
    if save_to_txt:
        save_txt(train_sensor_data, "train_sensor_data")
        save_txt(test_sensor_data, "test_sensor_data")
        save_txt(val_sensor_data, "val_sensor_data")
        save_txt(train_labels, "train_labels")
        save_txt(test_labels, "test_labels")
        save_txt(val_labels, "val_labels")

    train_labels = np.transpose(train_labels)
    test_labels = np.transpose(test_labels)
    val_labels = np.transpose(val_labels)
    train_labels = tf.expand_dims(train_labels, axis=-1)
    test_labels = tf.expand_dims(test_labels, axis=-1)
    val_labels = tf.expand_dims(val_labels, axis=-1)

    """Tensorflow Dataset"""
    train_data = tf.data.Dataset.from_tensor_slices(train_sensor_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_sensor_data)
    val_data = tf.data.Dataset.from_tensor_slices(val_sensor_data)
    train_label = tf.data.Dataset.from_tensor_slices(train_labels)
    test_label = tf.data.Dataset.from_tensor_slices(test_labels)
    val_label = tf.data.Dataset.from_tensor_slices(val_labels)
    
    """windowing"""
    train_data = train_data.window(size=250, shift=125, drop_remainder=True)
    train_data = train_data.flat_map(lambda window: window.batch(250))
    train_label = train_label.window(size=250, shift=125, drop_remainder=True)
    train_label = train_label.flat_map(lambda window: window.batch(250))
    train_ds = tf.data.Dataset.zip((train_data, train_label))

    val_data = val_data.window(size=250, shift=125, drop_remainder=True)
    val_data = val_data.flat_map(lambda window: window.batch(250))
    val_label = val_label.window(size=250, shift=125, drop_remainder=True)
    val_label = val_label.flat_map(lambda window: window.batch(250))
    val_ds = tf.data.Dataset.zip((val_data, val_label))

    test_data = test_data.window(size=250, shift=125, drop_remainder=True)
    test_data = test_data.flat_map(lambda window: window.batch(250))
    test_label = test_label.window(size=250, shift=125, drop_remainder=True)
    test_label = test_label.flat_map(lambda window: window.batch(250))
    test_ds = tf.data.Dataset.zip((test_data, test_label))

    return train_ds, val_ds, test_ds


def load_dataset():
    """Load dataset"""
    train_sensor_data = np.loadtxt("train_sensor_data")
    test_sensor_data = np.loadtxt("test_sensor_data")
    val_sensor_data = np.loadtxt("val_sensor_data")
    train_labels = np.loadtxt("train_labels")
    test_labels = np.loadtxt("test_labels")
    val_labels = np.loadtxt("val_labels")
    train_labels = np.transpose(train_labels)
    test_labels = np.transpose(test_labels)
    val_labels = np.transpose(val_labels)
    train_labels = tf.expand_dims(train_labels, axis=-1)
    test_labels = tf.expand_dims(test_labels, axis=-1)
    val_labels = tf.expand_dims(val_labels, axis=-1)

    """tf dataset"""
    train_data = tf.data.Dataset.from_tensor_slices(train_sensor_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_sensor_data)
    val_data = tf.data.Dataset.from_tensor_slices(val_sensor_data)
    train_label = tf.data.Dataset.from_tensor_slices(train_labels)
    test_label = tf.data.Dataset.from_tensor_slices(test_labels)
    val_label = tf.data.Dataset.from_tensor_slices(val_labels)

    """windowing"""
    train_data = train_data.window(size=250, shift=125, drop_remainder=True)
    train_data = train_data.flat_map(lambda window: window.batch(250))
    train_label = train_label.window(size=250, shift=125, drop_remainder=True)
    train_label = train_label.flat_map(lambda window: window.batch(250))
    train_ds = tf.data.Dataset.zip((train_data, train_label))

    val_data = val_data.window(size=250, shift=125, drop_remainder=True)
    val_data = val_data.flat_map(lambda window: window.batch(250))
    val_label = val_label.window(size=250, shift=125, drop_remainder=True)
    val_label = val_label.flat_map(lambda window: window.batch(250))
    val_ds = tf.data.Dataset.zip((val_data, val_label))

    test_data = test_data.window(size=250, shift=125, drop_remainder=True)
    test_data = test_data.flat_map(lambda window: window.batch(250))
    test_label = test_label.window(size=250, shift=125, drop_remainder=True)
    test_label = test_label.flat_map(lambda window: window.batch(250))
    test_ds = tf.data.Dataset.zip((test_data, test_label))

    return train_ds, val_ds, test_ds


def dataset_for_oversample(dataset):
    filtered_data = []
    filtered_labels = []

    for d, l in dataset:
        result = tf.reduce_any(tf.equal(l, tf.constant([7, 8, 9, 10, 11, 12], dtype=l.dtype)), axis=-1)
        if tf.reduce_any(result):
            filtered_data.append(d)
            filtered_labels.append(l)

    filtered_data = tf.data.Dataset.from_tensor_slices(filtered_data)
    filtered_labels = tf.data.Dataset.from_tensor_slices(filtered_labels)
    filtered_dataset = tf.data.Dataset.zip((filtered_data, filtered_labels))

    return filtered_dataset

def prepare(train_ds, val_ds, test_ds, batch_size=300 , repeats=0,std=0):
    print("repeats,std:",repeats,std)
    
    """augmentation including oversampling and adding noise"""
    train_filtered_dataset = dataset_for_oversample(train_ds)
    print('repeats=',repeats,'noise_std=',std)
    for i in range(0, repeats):
        train_ds = train_ds.concatenate(train_filtered_dataset)
    train_ds = train_ds.map(lambda data, label: aug(data, label, std=std), num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
    """Prepare training dataset"""
    train_ds = train_ds.shuffle(1500)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.repeat(-1)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    """Prepare validation dataset"""
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    """Prepare test dataset"""
    test_ds = test_ds.batch(923)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds, test_ds

