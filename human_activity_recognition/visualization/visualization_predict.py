"""
This script was developed by Shuaike Liu for visualizing predicted human activity on a specific user data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


colors = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4',
          '#f032e6', '#a9a9a9', '#808000', '#9A6324']

def visualization_pred():
    y_pred = np.loadtxt("y_pred")
    sensor_data = np.loadtxt("user22_data")
    sensor_data = sensor_data[:len(y_pred)]
    acc_data = sensor_data[:, :3]
    gyro_data = sensor_data[:, 3:]

    timestamps = np.arange(len(y_pred))
    labels = y_pred.astype(int)

    fig = plt.figure(figsize=(8, 4), dpi=500)

    # Create a grid with 2 rows and 1 column
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.2)

    # Main plot for sensor data and labels
    ax0 = plt.subplot(gs[0])
    for i in range(3):
        ax0.plot(timestamps, acc_data[:, i], label=f'acc_{i + 1}')
        ax0.plot(timestamps, gyro_data[:, i], label=f'gyro_{i + 1}')

    ax0.set_title('Predicted label for user22_exp44')
    ax0.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
    ax0.get_legend().remove()# Move legend outside the plot area

    # Highlight regions where the label is 0 with transparent background
    for target_label, color in zip(range(1, 13), colors):
        for timestamp, label in zip(timestamps, labels):
            if label == target_label:
                ax0.axvspan(timestamp, timestamp + 1, facecolor=color, alpha=0.3)

    # Color bar for labels
    ax1 = plt.subplot(gs[1])
    ax1.set_xticks(np.arange(0, 12, 1) + 1.5)
    ax1.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], fontsize=8)
    ax1.get_yaxis().set_visible(False)

    for i, color in zip(range(1, 13), colors):
        ax1.axvspan(i, i + 1, facecolor=color, alpha=0.31)

    ax1.set_xlabel('Labels')

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)

    # Show the plot
    plt.show()

visualization_pred()
