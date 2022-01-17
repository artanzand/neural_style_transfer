# author: Artan Zandian
# date: 2022-01-22

"""Reads two source images, one as the initial content image and second as the target style image,
and applies Neural Style Transfer on the content image to create a stylized rendering of the content
image based on the texture and style of the style image.

Usage: stylize.py --content=<image_path> --style=<csv_path> --save=<save_path> --similarity=<direction>

Options:
--content=<image_path>               file path of the content image - initial  
--style=<csv_path>                   file path of the style image - target
--save=<save_path>                   file path to save the stylized image 
--similarity=<direction>             Whether the generated image is similar to "content", "style", "balanced"
"""

import os
import sys
import scipy.io
import scipy.misc
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

from docopt import docopt


opt = docopt(__doc__)


def main(content, style, save):
    """
    
    Parameters
    ----------

    Returns
    -------
    """
    # Limit the image size to increase performance
    img_size = 400

    # Load pretrained VGG19 model
    vgg = tf.keras.applications.VGG19(
        include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet"
    )
    # Lock in the model weights
    vgg.trainable = False


    # Load Content and Style images
    content_image = preprocess_image(content)
    content_image = preprocess_image(style)
    
    # Randomly initialize Generated image
    # Setting the generated image as the variable to optimize
    generated_image = tf.Variable(
        tf.image.convert_image_dtype(content_image, tf.float32)
    )
    # Add random noise to initial generated image
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(
        generated_image, clip_value_min=0.0, clip_value_max=1.0
    )




def get_layer_outputs(vgg, layer_names):
    """ 
    Creates a vgg model that returns a list of intermediate output values.
    """
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def get_style_layers(similarity="balanced"):
    """
    Assigns weights to style layer outputs to define whether the generated image 
    is similar to "content", "style", or "balanced". The function is picking the
    last convolutional layer in each of the five blocks of the VGG network. The 
    activations of each of these layers along with the content layer (last layer)
    will be the outputs of the neural style transfer network.

    Parameters
    ----------
    similarity: str, optional
        a string identifying the similarity to either content, style or both
    
    Returns
    -------
    STYLE_LAYERS
        a list of tuples identifying the name of style layer along with their weights
    """
    if similarity == "balanced":
        STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
    elif similarity == "content":
        STYLE_LAYERS = [
        ('block1_conv1', 0.02),
        ('block2_conv1', 0.08),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.3),
        ('block5_conv1', 0.4)]
    elif similarity == "style":
        ('block1_conv1', 0.4),
        ('block2_conv1', 0.3),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.08),
        ('block5_conv1', 0.02)]
    else:
        raise Exception("Please provide either of 'content', 'style' or 'balanced' for --similarity")

    return STYLE_LAYERS


def preprocess_image(image_path):
    """
    loads the image and makes it compatible with VGG input size
    
    Parameters
    ----------
    image_path: str
        directory path of the image
    
    Returns
    -------
    image
        loaded and standardaized image
    """
    # Load Content and Style images
    # Resize both to a square image
    image = np.array(Image.open(image_path).resize((img_size, img_size)))
    # Add one dim for VGG compatibility
    image = tf.constant(np.reshape(image, ((1,) + image.shape)))

    return image



if __name__ == "__main__":
    main(opt["--content"], opt["--style"], opt["--save"], opt["--similarity"])
