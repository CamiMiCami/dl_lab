'''
This script was modified from original given script and further developed by Shuaike Liu to finish input pipeline. 
It includes two functions load() and prepare(). Both of them are called in main.py to generate dataset as model input.

The function load(name, data_dir, seed) is used to load dataset found by its name and dir.
These two parameters should be defined in gin configuration file.
Seed is used to seperate training set and validation set. It is configured in main.py by wandb

The function prepare(ds_train, ds_val, ds_test, ds_info, apply_flip, x_bright, x_contrast, x_saturation, x_quality, batch_size=15, caching=False)
is used to prepare dataset.
Parameters apply_flip, x_bright, x_contrast, x_saturation, x_quality are passed to function augment which is called here
'''
import gin
import gin.tf
import pandas as pd
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import random

from input_pipeline.preprocessing import preprocess, augment


@gin.configurable
def load(name, data_dir, seed=0):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Set the path
        train_image_dir = data_dir + "/images/train"
        test_image_dir = data_dir + "/images/test"
        train_labels_file = data_dir + "/labels/train.csv"
        test_labels_file = data_dir + "/labels/test.csv"

        # Import raw data from dir
        train_csv = pd.read_csv(train_labels_file)
        test_csv = pd.read_csv(test_labels_file)
        train_image_names = sorted(glob.glob(train_image_dir + "/*.jpg"))
        test_image_names = sorted(glob.glob(test_image_dir + "/*.jpg"))
        train_labels = list(train_csv["Retinopathy grade"])
        test_labels = list(test_csv["Retinopathy grade"])
        

        # Make training labels binary
        for index, label in enumerate(train_labels):
            if label > 1:
                train_labels[index] = 1
            else:
                train_labels[index] = 0
        # Make test labels binary
        for index, label in enumerate(test_labels):
            if label > 1:
                test_labels[index] = 1
            else:
                test_labels[index] = 0
                

                

        # Read and decode images
        def read_image(file_name, label):
            image_string = tf.io.read_file(file_name)
            image_decoded = tf.io.decode_jpeg(image_string)
            return image_decoded, label

        # Create a dataset
        def create_dataset(file_name, label):
            ds = tf.data.Dataset.from_tensor_slices((file_name, label))
            ds = ds.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return ds
        
        num_samples = len(train_image_names)
        num_train = int(0.8 * num_samples)
        num_val = num_samples - num_train
        
        if seed != 0:
            random.seed(seed)
        random_indices = random.sample(range(num_samples), num_train)
        
        #split training set and validation set
        ds_train = create_dataset([train_image_names[i] for i in random_indices], [train_labels[i] for i in random_indices])
        
        #check the distribution in training dataset
        num_label_0 = sum(1 for _, label in ds_train.as_numpy_iterator() if label == 0)
        num_label_1 = sum(1 for _, label in ds_train.as_numpy_iterator() if label == 1)
        print("Number of training samples with label 0:", num_label_0, num_label_0/num_train)
        print("Number of training samples with label 1:", num_label_1, num_label_1/num_train)
        
        ds_val = create_dataset([train_image_names[i] for i in range(num_samples) if i not in random_indices], 
                        [train_labels[i] for i in range(num_samples) if i not in random_indices])
        
        #check the distribution in training dataset
        num_label_0 = sum(1 for _, label in ds_val.as_numpy_iterator() if label == 0)
        num_label_1 = sum(1 for _, label in ds_val.as_numpy_iterator() if label == 1)
        print("Number of validation samples with label 0:", num_label_0, num_label_0/num_val)
        print("Number of validation samples with label 1:", num_label_1, num_label_1/num_val)

        print(f"Number of training samples: {num_train}") 
        print(f"Number of validation samples: {num_val}")
        print("Number of test samples:", len(test_image_names))
        
        ds_test = create_dataset(file_name=test_image_names, label=test_labels)
        ds_info = None
        return ds_train, ds_val, ds_test, ds_info  # type: ignore

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "diabetic_retinopathy_detection/btgraham-300",
            split=["train", "validation", "test"],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir,
        )  # type: ignore

        def _preprocess(img_label_dict):
            return img_label_dict["image"], img_label_dict["label"]

        ds_train = ds_train.map(
            _preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_val = ds_val.map(
            _preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_test = ds_test.map(
            _preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return ds_train, ds_val, ds_test, ds_info  # type: ignore

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist",
            split=["train[:90%]", "train[90%:]", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir,
        )  # type: ignore

        return ds_train, ds_val, ds_test, ds_info  # type: ignore

    else:
        raise ValueError


def prepare(ds_train, ds_val, ds_test, ds_info, apply_flip, x_bright, x_contrast, x_saturation, x_quality, batch_size=15, caching=False):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(lambda image, label: augment(image, label, apply_flip, x_bright, x_contrast, x_saturation, x_quality), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(330 // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
