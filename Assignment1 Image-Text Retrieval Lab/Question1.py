import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import numpy
%matplotlib inline

# Load data and show some images.
data = pickle.load(open('mscoco_small.p'))
train_data = data['train']
val_data = data['val']

# Pick an image and show the image.
sampleImageIndex = 310 # Try changing this number and visualizing some other images from the dataset.
plt.figure()
plt.imshow(imread('mscoco/%s' % train_data['images'][sampleImageIndex]))
print(sampleImageIndex, train_data['captions'][sampleImageIndex])