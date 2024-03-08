"""
This script was developed by Shuaike Liu for
Creating a data file and a label file for a specific user experiment pair for evaluation purpose
Things need to be changed for different users and experiments:
- acc_pattern
- gyro_pattern
- exp_value
- user_value
"""
import tensorflow as tf
import numpy as np
import os
import re


def save_txt(array, file_path):
    np.savetxt(file_path, array)


def z_score(array):
    mean_values = np.mean(array, axis=0)
    std_dev_values = np.std(array, axis=0)

    return (array - mean_values) / std_dev_values


def load(save_to_txt=True):
    file_raw = r"C:\Users\shuai\Desktop\smartphone+based+recognition+of+human+activities+and+postural+transitions\RawData"
    file_label = r"C:\Users\shuai\Desktop\smartphone+based+recognition+of+human+activities+and+postural+transitions\RawData\labels.txt"

    acc_ds = np.empty((0, 6))
    gyro_ds = np.empty((0, 3))
    labels = np.loadtxt(file_label)
    exp = []
    user = []

    acc_pattern = r"^acc_exp44_user22.txt$"  # change
    gyro_pattern = r"^gyro_exp44_user22.txt$"  # change

    for filename in sorted(os.listdir(file_raw)):
        acc_match = re.match(acc_pattern, filename)
        gyro_match = re.match(gyro_pattern, filename)

        if acc_match:
            data = np.loadtxt(os.path.join(file_raw, filename))
            row_indices = np.arange(1, data.shape[0] + 1)[:, np.newaxis]
            data = np.concatenate((row_indices, data), axis=1)
            exp_value = 44  # change
            user_value = 22  # change
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

    # Remove exp, user and time column from *_sensor_data and label
    sensor_data = sensor_data[:, 3:]

    # Z-Score Normalization
    sensor_data = z_score(sensor_data)

    # Save files
    if save_to_txt:
        save_txt(sensor_data, "user22_data")
        save_txt(labels_ds, "user22_labels")

