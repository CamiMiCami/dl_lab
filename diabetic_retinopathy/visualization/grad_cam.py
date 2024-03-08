"""
This script was developed by Shuaike Liu.
It has two classes GradCam and GuidedGradCam, which are used to visualize the data.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Class for generating Grad-CAM visualizations
class GradCam:
    def __init__(self, model, last_conv_layer_name, pred_index=None):
        self.model = model
        self.pred_index = pred_index
        self.last_conv_layer_name = last_conv_layer_name

    # Generate Grad-CAM heatmap
    def make_gradcam_heatmap(self, image, n_blocks):
        # Get last conv layer name
        if self.last_conv_layer_name is None:
            self.last_conv_layer_name = "conv2d_"+str((n_blocks*2)-1)

        # Gradient model with the desired activation layer
        grad_model = tf.keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        # Remove it since only one neuron at output
        grad_model.layers[-1].activation = None

        # Record gradients
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image)
            print(preds)
            if self.pred_index is None:
                self.pred_index = tf.argmax(preds[0])
            class_channel = preds[:, self.pred_index]
            print(class_channel)

        # Compute gradients of the predicted class with respect to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute the heatmap
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        return heatmap

    # Method to visualize Grad-CAM heatmap overlaid on the original image
    def show_gradcam(self, image_path, heatmap, alpha=0.4):
        image = tf.keras.utils.load_img(image_path)
        image = tf.keras.utils.img_to_array(image)

        # Generate heatmap overlaid on the image
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * alpha + image
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

        # Create subplots
        plt.matshow(superimposed_img)
        plt.axis('off')
        plt.show()


# Guided grad cam
class GuidedGradCam(GradCam):
    def __init__(self, model, last_conv_layer_name, pred_index=None):
        super().__init__(model, last_conv_layer_name, pred_index)

    def make_guided_gradcam_heatmap(self, image, n_blocks):
        if self.last_conv_layer_name is None:
            self.last_conv_layer_name = "conv2d_"+str((n_blocks*2)-1)

        grad_model = tf.keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        grad_model.layers[-1].activation = None

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image)
            print(preds)
            if self.pred_index is None:
                self.pred_index = tf.argmax(preds[0])
            class_channel = preds[:, self.pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]

        # Guided backpropagation
        guided_grads = tf.cast(last_conv_layer_output > 0, 'float32') * tf.cast(pooled_grads > 0,
                                                                                'float32') * pooled_grads
        # Compute Guided Grad-CAM heatmap
        heatmap = tf.reduce_mean(guided_grads * last_conv_layer_output, axis=-1)
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        return heatmap


def load_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(255, 255))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32) / 255.0
    return img_array