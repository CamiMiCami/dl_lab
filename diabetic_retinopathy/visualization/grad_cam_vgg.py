"""
This script was developed by Shuaike Liu for visualizing the prediction of diabetic retinopathy from VGG.
"""
import tensorflow as tf
from architectures import vgg_like
import os
import grad_cam
import numpy as np

# Because of Intel bug
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


model = vgg_like(base_filters=8, n_blocks=4, dense_units=32, dropout_rate=0.3)
checkpoint = 'ckpt-194'
status = tf.train.Checkpoint(model=model).restore(checkpoint)
status.expect_partial()
model.summary()

# Your image path
img_path = r"C:\Users\shuai\Desktop\uni\dl-lab-23w-team15\IDRID_dataset\images\test\IDRiD_033.jpg"
img_array = grad_cam.load_image(img_path)

grad = grad_cam.GradCam(model, last_conv_layer_name="conv2d_7")

heatmap = grad.make_gradcam_heatmap(image=img_array, n_blocks=4)
grad.show_gradcam(image_path=img_path, heatmap=heatmap)
