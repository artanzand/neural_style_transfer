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







if __name__ == "__main__":
    main(opt["--content"], opt["--style"], opt["--save"])