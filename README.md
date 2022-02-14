# neural_style_transfer [![Build Status](https://github.com/anishathalye/neural-style/workflows/CI/badge.svg)](https://github.com/anishathalye/neural-style/actions?query=workflow%3ACI)
Author: Artan Zandian  
Date: January 2022

## About
This project is a Tensorflow implementation of unsupervised deep learning algorithm originally introduced by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576). For review of the framework and to understand how each individual piece works please refer to [my project post](https://artanzand.github.io//neural-style-transfer/).
<p align="center">
  <img src="https://github.com/artanzand/neural_style_transfer/blob/main/examples/balloon_style.gif" />
</p>

## Usage
### Cloning the Repo
Clone this Github repository and install the dependencies by running the following commands at the command line/terminal from the root directory of the project:

```conda env create --file environment.yaml```  
```conda activate NST```

Run the below command by replacing content image , style image and save directory.  
```python stylize.py --content <content image> --style <style image> --save <save directory>```

Run `python stylize.py --help` to see a list of all options and input type details.  
  
Two optional arguments of --similarity (default "balanced") and --epochs (default 500) control the similarity to either of input photos and number of iternations respectively. 
For a 512×680 pixel content file, 1000 iterations take 75 seconds on an Nvidia Jetson Nano 2GB, or 75 minutes on an Intel Core i5-8250U. Due to the speedup using a GPU is highly recommended.

### Using NVIDIA Docker  
Do you have access to an NVIDIA Jetson device? I have created an NVIDIA docker image which can be pulled to your device (I have tested it on Jetson Nano) using the commands below. If you are interensted in learning how to build a custom NVIDIA image using Podman, check out my blog [here](https://artanzand.github.io//Tensorflow-Docker/).
  
docker run artanzandian/keras:0.1.0 --rm -it 

1. clone the repository  
2. change directory to the root of the project  
```sudo docker run --rm -it -v /$(pwd):/home/ artanzandian/keras:0.1.0```  
3. Interactive usage:  
```
cd home
python stylize.py --content=examples/balloon.JPG --style=examples/city-rain.jpg --save=examples/stylized --similarity=style --epochs=2000
```



## Examples
Running for 1000-2500 epochs usually produce nice results. If the light and exposure of the content and style images are similar, smaller number of epochs would work better while the model seems to be overfitting if the number of epochs is increases to 10,000. Overfitting in case of this model could be described as too much abstraction where the overall shape of the content photo is lost. The following example was run for 2500 iterations to produce the result (with default parameters). The style input image was Bob Ross's famous Summer painting, and the content image was my own image from Moraine Lake, Alberta.
<p align="center">
  <img src="https://github.com/artanzand/neural_style_transfer/blob/main/examples/moraine_style.JPG" />
</p>
<br>

In the next example due to having a different light exposure (day vs night) the number of epochs were increased to 20,000 to generate decent looking results. The style input image was Rainy Night in the City by Leonid Afremov, and the content image was my own image from a balloon ride in Calgary, Alberta.
<p align="center">
  <img src="https://github.com/artanzand/neural_style_transfer/blob/main/examples/all-three.JPG" />
</p>
<br>


## Parameters
### Epochs
As briefly alluded to in the Examples section, number of epochs controls the number of iterations and the proper size of epochs is highly dependent on the two input images.
<p align="center">
  <img src="https://github.com/artanzand/neural_style_transfer/blob/main/examples/epochs.JPG" />
</p>
<br>

### Similarity
There are five layers within the base VGG19 architecture used for this project. The weight of these layers determine how similar the output image will be to either of the two input images. For implementaiton details see [my post](https://artanzand.github.io//neural-style-transfer/). Instead of giving the user full selection freedom over the layer weights which are very confusing and hard to control, I have created three optimized options of layer weights which can be changed based on user preference to have an image that has similar details to the "content" image, "style" image or is "balanced" between the two. Below shows 20,000 iterations for each of the similarity options.   

<p align="center">
  <img src="https://github.com/artanzand/neural_style_transfer/blob/main/examples/similarity.JPG" />
</p>
<br>


## Requirements

### Neural Network Model and Weights
The main function in `stylize.py` loads [VGG19 Architecture](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19) from Keras with ImageNet weights.


### Dependencies  

A complete list of dependencies is available
[here](https://github.com/artanzand/neural_style_transfer/blob/main/environment.yaml).
<br>- python=3.9
<br>- tensorflow=2.6
<br>- keras=2.6
<br>- docopt=0.6.1
<br>- pillow=8.4.0


## License
This project is licensed under the terms of the MIT license.


## Credits and References
Gatys, Leon A., Ecker, Alexander S. and Bethge, Matthias. "A Neural Algorithm of Artistic Style.." CoRR abs/1508.06576 (2015): [link to paper](https://arxiv.org/abs/1508.06576)    
Athalye A., athalye2015neuralstyle, "Neural Style" (2015): [Repository](https://github.com/anishathalye/neural-style)   

[DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes 
