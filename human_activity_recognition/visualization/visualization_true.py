"""
This script was developed by Shuaike Liu for visualizing true human activity on a specific user data.
"""
import numpy as np
import matplotlib.pyplot as plt
import prepare_visu_data

prepare_visu_data.load()  # Creating data and label files in the same directory

colors = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4',
          '#f032e6', '#a9a9a9', '#808000', '#9A6324']


def visualization_true():
    # Load sensor data
    sensor_data = np.loadtxt(r"user22_data")
    acc_data = sensor_data[:, :3]
    gyro_data = sensor_data[:, 3:]

    # Load label data
    y_true = np.loadtxt(r"user22_labels")

    # Extract timestamps and labels
    timestamps = np.arange(len(y_true))
    labels = y_true.astype(int)

    fig, ax = plt.subplots(figsize=(8, 2), dpi=500)

    # Plot accelerometer data
    for i in range(3):
        ax.plot(timestamps, acc_data[:, i], label=f'acc_{i+1}')

    # Plot gyroscope data
    for i in range(3):
        ax.plot(timestamps, gyro_data[:, i], label=f'gyro_{i+1}')

    ax.set_title('True label for user22_exp44')
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))  # Move legend outside the plot area
    ax.get_legend().remove()

    # Highlight regions where the label is 0 with transparent background
    for target_label, color in zip(range(1, 13), colors):
        for timestamp, label in zip(timestamps, labels):
            if label == target_label:
                ax.axvspan(timestamp, timestamp + 1, facecolor=color, alpha=0.3)

    # Show the plot
    plt.show()

visualization_true()