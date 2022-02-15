# author: Artan Zandian
# date: 2022-01-22

"""
Reads two source images, one as the initial content image and second as the target style image,
and applies Neural Style Transfer on the content image to create a stylized rendering of the content
image based on the texture and style of the style image.
Usage: python stylize.py --content <content image> --style <style image> --save <save directory> --similarity <direction> --epochs <num_iter>
Options:
--content=<image_path>               file path of the content image - initial  
--style=<csv_path>                   file path of the style image - target
--save=<save_path>                   file path to save the stylized image without image format
--similarity=<direction>             Whether the generated image is similar to "content", "style", "balanced"
--epochs=<num_iter>                  number of epochs - 2,000 for speed, 10,000 for quality
"""


from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

from docopt import docopt

opt = docopt(__doc__)


def main(content, style, save, similarity="balanced", epochs=500):
    """
    The main function reads two source images, one as the initial content image
    and second as the target style image, and applies Neural Style Transfer on
    the content image to create a stylized rendering of the content image based on
    the texture and style of the style image.
    Parameters
    ----------
    content: str
        The image path to the content image to start from
    style: str
        The image path to the target style image
    save: str
        The path to save the image without image type
    similarity: str, optional
        whether the generate image is similar to 'content', 'style' or 'balanced'
    epochs: int, optional
        number of iterations to train the generate image.
    Returns
    -------
    image
        saved stylized image
    """
    # Exception handelings
    try:
        type(int(epochs)) == int
    except Exception:
        raise ("epochs should be an integer value!")

    try:
        # Limit the image size to increase performance
        image_size = 400

        # capture content image size to reshape at end
        content_image = Image.open(content)
        content_width, content_height = content_image.size

        # Load pretrained VGG19 model
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            input_shape=(image_size, image_size, 3),
            weights="imagenet",
        )
        # Lock in the model weights
        vgg.trainable = False

        # Load Content and Style images
        content_image = preprocess_image(content, image_size)
        style_image = preprocess_image(style, image_size)

        # Randomly initialize Generated image
        # Define the generated image as as tensorflow variable to optimize
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
        style_layers = get_style_layers(similarity=similarity)
        content_layer = [("block5_conv4", 1)]  # The last layer of VGG19

        vgg_model_outputs = get_layer_outputs(vgg, style_layers + content_layer)

        # Content encoder
        # Define activation encoding for the content image (a_C)
        # Assign content image as the input of VGG19
        preprocessed_content = tf.Variable(
            tf.image.convert_image_dtype(content_image, tf.float32)
        )
        a_C = vgg_model_outputs(preprocessed_content)

        # Style encoder
        # Define activation encoding for the style image (a_S)
        # Assign style image as the input of VGG19
        preprocessed_style = tf.Variable(
            tf.image.convert_image_dtype(style_image, tf.float32)
        )
        a_S = vgg_model_outputs(preprocessed_style)

        # Initialize the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # Need to redefine the clipped image as a tf.variable to be optimized
        generated_image = tf.Variable(generated_image)

        # Check if GPU is available
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

        # Train the model
        epochs = int(epochs)
        for i in range(epochs):
            train_step(
                generated_image, vgg_model_outputs, style_layers, optimizer, a_C, a_S
            )
            if i % 500 == 0:
                print(f"Epoch {i} >>>")

        # Resize to original size and save
        image = tensor_to_image(generated_image)
        image = image.resize((content_width, content_height))
        image.save(save + ".jpg")
        print("Image saved.")

    except Exception as message:
        print(message)


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
    style_layers
        a list of tuples identifying the name of style layer along with their weights
    """
    if similarity == "balanced":
        style_layers = [
            ("block1_conv1", 0.2),
            ("block2_conv1", 0.2),
            ("block3_conv1", 0.2),
            ("block4_conv1", 0.2),
            ("block5_conv1", 0.2),
        ]
    elif similarity == "content":
        style_layers = [
            ("block1_conv1", 0.02),
            ("block2_conv1", 0.08),
            ("block3_conv1", 0.2),
            ("block4_conv1", 0.3),
            ("block5_conv1", 0.4),
        ]
    elif similarity == "style":
        style_layers = [
            ("block1_conv1", 0.4),
            ("block2_conv1", 0.3),
            ("block3_conv1", 0.2),
            ("block4_conv1", 0.08),
            ("block5_conv1", 0.02),
        ]
    else:
        raise Exception(
            "Please provide either of 'content', 'style' or 'balanced' for --similarity"
        )

    return style_layers


def preprocess_image(image_path, image_size):
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
    # Load and resize Content and Style images to a square image
    image = np.array(Image.open(image_path).resize((image_size, image_size)))
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
        tensor = tensor[0]
    return Image.fromarray(tensor)


@tf.function()
def train_step(generated_image, vgg_model_outputs, style_layers, optimizer, a_C, a_S):
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
        J_style = compute_style_cost(a_S, a_G, style_layers)

        # Compute total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(
        tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    )


def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost.
    Parameters
    ----------
    a_C: tensor
        hidden layer activations representing content of the image C - dimension (1, n_H, n_W, n_C)
    a_G: tensor
        hidden layer activations representing content of the image G - dimension (1, n_H, n_W, n_C)
    Returns
    -------
    J_content: float64
        the content cost between a_C and a_G
    """
    # Exclude the last layer output
    a_C = content_output[-1]
    a_G = generated_output[-1]

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, shape=(1, -1, n_C))
    a_G_unrolled = tf.reshape(a_G, shape=(1, -1, n_C))

    # compute the cost with tensorflow
    J_content = (1 / (4 * n_C * n_H * n_W)) * tf.reduce_sum(
        tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))
    )

    return J_content


def compute_layer_style_cost(a_S, a_G):
    """
    Computes the style cost of one layer.
    Parameters
    ----------
    a_C: tensor
        hidden layer activations representing content of the image C - dimension (1, n_H, n_W, n_C)
    a_G: tensor
        hidden layer activations representing content of the image G - dimension (1, n_H, n_W, n_C)
    Returns
    -------
    J_style_layer
        A scalar value representing style cost for a layer
    """

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images from (1, n_H, n_W, n_C) to have them of shape (n_C, n_H * n_W)
    a_S = tf.reshape(tf.transpose(a_S, perm=[3, 0, 1, 2]), shape=(n_C, -1))
    a_G = tf.reshape(tf.transpose(a_G, perm=[3, 0, 1, 2]), shape=(n_C, -1))

    # Computing gram_matrices for both images S and G
    GS = tf.matmul(a_S, tf.transpose(a_S))
    GG = tf.matmul(a_G, tf.transpose(a_G))

    # Computing the loss
    J_style_layer = (1 / (2 * n_C * n_H * n_W) ** 2) * tf.reduce_sum(
        tf.square(tf.subtract(GS, GG))
    )

    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, style_layers):
    """
    Computes the overall style cost from several chosen layers
    Parameters
    ----------
    style_image_output: tensor
        output of VGG model for the style image (activations of style layers & content layer)
    generated_image_output: tensor
        output of VGG model for the generated image (activations of style layers & content layer)
    style_layers : list of tuples
        containing the names of the layers we would like to extract style from and a coefficient for each of them
    Returns
    -------
    J_style
        A scalar value representing style cost
    """

    # initialize the cost
    J_style = 0

    # Excluding the last element of the array which contains the content layer image
    a_S = style_image_output[:-1]  # a_S is the hidden layer activations
    a_G = generated_image_output[:-1]  # a_G is the hidden layer activations

    for i, weight in zip(range(len(a_S)), style_layers):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function. Because the main purpose of the algorithm
    is on matching the style of a target photo a bigger weight (beta) is given to
    the style image.
    Parameters
    ----------
    J_content: float
        content cost computed in compute_content_cost
    J_style: float
        style cost computed in compute_style_cost
    alpha: float
        hyperparameter weighting the importance of the content cost
    beta: float
        hyperparameter weighting the importance of the style cost
    Returns
    -------
    J
        total cost
    """
    J = alpha * J_content + beta * J_style

    return J


if __name__ == "__main__":
    main(
        opt["--content"],
        opt["--style"],
        opt["--save"],
        opt["--similarity"],
        opt["--epochs"],
    )
