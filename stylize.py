# author: Artan Zandian
# date: 2022-01-22

"""Reads two source images, one as the initial content image and second as the target style image,
and applies Neural Style Transfer on the content image to create a stylized rendering of the content
image based on the texture and style of the style image.

Usage: stylize.py --content=<image_path> --style=<csv_path> --save=<save_path>

Options:
--content=<image_path>               file path of the content image - initial  
--style=<csv_path>                   file path of the style image - target
--save=<save_path>                   file path to save the stylized image 
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

    # Limit the image size to increase performance
    img_size = 400

    # Load pretrained VGG19 model
    vgg = tf.keras.applications.VGG19(
        include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet"
    )
    # Lock in the model weights
    vgg.trainable = False

    # Load content and style images
    # Resize both to a square image
    content_image = np.array(Image.open(content).resize((img_size, img_size)))
    # Add one dim for VGG compatibility
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

    style_image = np.array(Image.open(style).resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    # Initialize generated image
    generated_image = tf.Variable(
        tf.image.convert_image_dtype(content_image, tf.float32)
    )
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(
        generated_image, clip_value_min=0.0, clip_value_max=1.0
    )


if __name__ == "__main__":
    main(opt["--content"], opt["--style"], opt["--save"])
