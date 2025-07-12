## Black and white image colorization using Python, OpenCV and Deep Learning

![person](https://pyimagesearch.com/wp-content/uploads/2019/02/bw_colorization_opencv_robin_williams.jpg)

### Overview
This script is designed to colorize a black-and-white (grayscale) image using a pre-trained deep learning model. 
The model was originally created by researchers at UC Berkeley and has been made available for public use. 
The script uses OpenCV's deep neural network (DNN) module to perform the colorization task.

## Download these files in advance if not setup from rep install

Download the model files:
1. colorization_deploy_v2.prototxt:[Protoxt file](https://github.com/richzhang/colorization/tree/caffe/colorization/models)
  
2. pts_in_hull.npy: [numpyfile](https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy)
  
3. colorization_release_v2.caffemodel: [caffemodel file](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1ZCSF9Kbjl4SXFjWWZiM0FZLV9xVW9iSjBwd3xBQ3Jtc0ttWHBUckdUb01Sd1ZueVJVTzNwd1htM0Y4X3I2OF9IVENyMk5laDVKWlBYWmZ1NjFYU2NJbGczeWp3QU9zS3JGbDJhc3BDa1RiZ1JqNm8zb0VCb0wtOE9KYWprV2ZTRzNYT2lBSVpGLXBucWstUS1kUQ&q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fdx0qvhhp5hbcx7z%2Fcolorization_release_v2.caffemodel%3Fdl%3D1&v=gAmskBNz_Vc)

## Steps to deploy

#### Prerequirements

Active version of Python

Git Version Control

PIP package installer


#### Download the repo as a zip file or run

`git clone https://github.com/lhcee3/Image-Colorization-DL.git`

#### Import the packages by running the command in cli

`pip install opencv-python numpy argparse os`

To run the colorization , I have sampled 4 images in the images folder.
You can execute it using the commands

`python color.py --image images/1.jpg`

`python color.py --image images/2.jpg`

`python color.py --image images/3.jpg`

`python color.py --image images/4.jpg`

#### Feel Free to Contribute !

Author : Rishi Prasad Sahani

