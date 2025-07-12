#Initial code
#input and output - terminal
'''
import numpy as np
import argparse
import cv2
import os 

#Orginal credits : https://richzhang.github.io/colorization/
#paths to load the models

# Set DIR to the parent directory containing the SAR directory
DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/model'))

# Correct paths to the model files
PROTOTEXT = os.path.join(DIR, r"colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"pts_in_hull.npy")
MODEL = os.path.join(DIR, r"colorization_release_v2.caffemodel")

# Command-line argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help=r"C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/images/1.jpg")
args = vars(ap.parse_args())

# Load the model
print("Load Model")
net = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)
pts = np.load(POINTS)


# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)


#converting LAB to RGB
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)

'''


#code with UI


import numpy as np
import cv2
import os
from PIL import Image

# ========== Load the model one time ==========
# Set DIR to the model folder
DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/model'))

# Correct paths to the model files
PROTOTEXT = os.path.join(DIR, r"colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"pts_in_hull.npy")
MODEL = os.path.join(DIR, r"colorization_release_v2.caffemodel")

# Load the model
print("Loading Model...")
net = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)
pts = np.load(POINTS)

# Setup network layers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


#UI-2

# Function for image colorization (ensure it's defined before)
def colorize_uploaded_image(input_pil_image):
    input_image = np.array(input_pil_image)

    # Check if grayscale (1 channel) and convert to RGB (3 channels)
    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

    scaled = input_image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Load model and colorize image (using your existing model loading code)
    net = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (input_image.shape[1], input_image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    output_pil_image = Image.fromarray(colorized)
    return output_pil_image
