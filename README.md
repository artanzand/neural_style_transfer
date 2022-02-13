# neural_style_transfer [![Build Status](https://github.com/anishathalye/neural-style/workflows/CI/badge.svg)](https://github.com/anishathalye/neural-style/actions?query=workflow%3ACI)
Author: Artan Zandian  
Date: January 2022

## About
This project is a Tensorflow implementation of unsupervised deep learning algorithm originally introduced by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576). For review of the framework and to understand how each individual piece works please refer to [my project blog](https://artanzand.github.io//neural-style-transfer/).
<p align="center">
  <img src="https://github.com/artanzand/neural_style_transfer/blob/main/examples/balloon_style.gif" />
</p>

## Usage
### Cloning the Repo
Clone this Github repository and install the dependencies by running the following commands at the command line/terminal from the root directory of the project:

`conda env create --file environment.yaml`
`conda activate NST`

Run the below command by replacing <content image>, <style image> and <save directory>.
`python stylize.py --content <content image> --style <style image> --save <save directory> --similarity <balanced> --epochs <num epochs>`

Run `python stylize.py --help` to see a list of all options and input type details.  
  
Two optional arguments of --similarity (default "balanced") and --epochs (default 500) control the similarity to either of input photos and number of iternations respectively. 
For a 512×680 pixel content file, 1000 iterations take 75 seconds on an Nvidia Jetson Nano 2GB, or 75 minutes on an Intel Core i5-8250U. Due to the speedup using a GPU is highly recommended.

### Using NVIDIA Docker  
Do you have access to an NVIDIA Jetson product? 
docker run docker.io/artanzandian/keras:0.1.0 




## Example




## Parameters




## Requirements

### Neural Network Model and Weights



### Dependencies  

A complete list of dependencies is available
[here](https://github.com/artanzand/neural_style_transfer/blob/main/environment.yaml).
<br>- python=3.9
<br>- tensorflow=2.6
<br>- keras=2.6
<br>- docopt=0.6.1
<br>- pillow=8.4.0


## Related Project


## License



## References
Gatys, Leon A., Ecker, Alexander S. and Bethge, Matthias. "A Neural Algorithm of Artistic Style.." CoRR abs/1508.06576 (2015): [link to paper](https://arxiv.org/abs/1508.06576)    
Athalye A., athalye2015neuralstyle, "Neural Style" (2015): [Repository](https://github.com/anishathalye/neural-style)   

[DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes 
