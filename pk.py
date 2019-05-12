#!/usr/bin/env python3
# Import some packages from matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# Uncomment the next line for use in a Jupyter notebook
#%matplotlib inline

# Define the filename, read and plot the image
filename = 'sample.jpg'
img0 = mpimg.imread(filename)
img = img0.copy()
img2 = img0.copy()
# Import the "numpy" package for working with arrays

rgb_thresh = (155,155,155);
R_THD = rgb_thresh[0]
G_THD = rgb_thresh[1]
B_THD = rgb_thresh[2]


r = img[:,:,0] #R channel 
g = img[:,:,1] #G channel 
b = img[:,:,2] #B channel 

r[img[:,:,0] < R_THD] = [0]
r[img[:,:,0] < (R_THD+1)] = [255]

g[img[:,:,1] < G_THD] = [0]
g[img[:,:,1] < (G_THD+1)] = [255]

b[img[:,:,2] < B_THD] = [0]
b[img[:,:,2] < (B_THD+1)] = [255]
# t = img[:,:,0] #R channel threshold
# thres = img[:,:,0] < 155
# thres1 = img[:,:,0] > 145

# t[thres] = [0]
# t[thres1] = [255]

plt.imshow(img)
plt.show()

#print("COLOR_SELECT:")
#print(img[:,:,0])
#color_select[thres] = 255
