'''
This script was developed by Shuaike Liu which includs 3 functions 
Function z_score() and function aug() was imported by datasets.py to make z-score normalization and adding noise to samples.
Function add_noise() is called by function aug() to calculate noise 

'''
import numpy as np
import tensorflow as tf
import random


def z_score(array):
    mean_values = np.mean(array, axis=0)
    std_dev_values = np.std(array, axis=0)

    return (array - mean_values) / std_dev_values


def aug(data, label, std):
    specific_numbers = np.array([7, 8, 9, 10, 11, 12])
    specific_numbers = tf.reshape(specific_numbers, [1, 6])
    data = tf.cast(data, tf.float64)
    label = tf.cast(label, tf.float64)

    contains_specific_numbers = tf.reduce_any(tf.equal(label, tf.cast(specific_numbers, tf.float64)), axis=-1)

    """tf.where is used to conditionally apply add_noise(data) 
    to the elements of data where the corresponding element in contains_specific_numbers is true.
    If contains_specific_numbers is true for a certain element, 
    add_noise(data) is applied; otherwise, the original value of data is used.
    So the second mode of tf.where is being used here"""
    data = tf.where(tf.expand_dims(contains_specific_numbers, axis=-1), add_noise(data, std=std), data)

    return data, label


def add_noise(data, mean=0.0, std=0.1):
    noise = tf.random.normal(shape=tf.shape(data), mean=mean, stddev=std, dtype=tf.float64)
    return data + noise

# oversample minors ---> !! tf.data.Dataset.sample_from_datasets
# hyperparameter ---> oversampling
