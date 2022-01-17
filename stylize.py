# author: Artan Zandian
# date: 2022-01-22

"""Reads two source images, one as the initial content image and second as the target style image,
and applies Neural Style Transfer on the content image to create a stylized rendering of the content
image based on the texture and style of the style image.

Usage: stylize.py --content=<image_path> --style=<csv_path> --save=<save_path> --similarity=<direction> --epochs=<num_iter>

Options:
--content=<image_path>               file path of the content image - initial  
--style=<csv_path>                   file path of the style image - target
--save=<save_path>                   file path to save the stylized image 
--similarity=<direction>             Whether the generated image is similar to "content", "style", "balanced"
--epochs=<num_iter>                  number of epochs - 5,000 for speed, 20,000 for quality
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


def main(content, style, save, similarity='balanced', epochs=500):
    """
    
    Parameters
    ----------

    Returns
    -------
    image
        saved stylized image
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


    # Define output layers
    STYLE_LAYERS = get_style_layers(similarity=similarity)
    content_layer = [('block5_conv4', 1)]  # The last layer of VGG19

    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    # Content encoder
    # Define activation encoding for the content image (a_C)
    # Assign content image as the input of the VGG19
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    # Style encoder
    # Define activation encoding for the style image (a_S)
    # Assign style image as the input of the VGG19
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Train the model
    epochs = epochs
    for i in range(epochs):
        train_step(generated_image)
        if i % 250 == 0:
            print(f"Epoch {i} >>>")
        
    image = tensor_to_image(generated_image)
    image.save(f"output/image_{i}.jpg")



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


def tensor_to_image(tensor):
    """
    Converts the calculated final vector into a PIL image
    
    Parameters
    ----------
    tensor: Tensor
    
    Returns
    -------
    Image
        A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


@tf.function()
def train_step(generated_image):
    """ 
    Uses precomputed encoded images a_S and a_C as constants, calculates 
    a_G as the encoding of the newly generated image, and uses the three
    to compute the cost function, and respectively, one gradient step.

    Parameters
    ----------
    generated_image: tensor
        image in shape of a vector
    """
    with tf.GradientTape() as tape:
        
        # a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        
        # Compute content cost
        J_content = compute_content_cost(a_C, a_G)

        # Compute style cost
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)

        # Compute total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)
        
        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))



if __name__ == "__main__":
    main(opt["--content"], opt["--style"], opt["--save"], opt["--similarity"], opt["--epochs"])
