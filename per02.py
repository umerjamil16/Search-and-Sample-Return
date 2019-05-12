import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2

# Read in the image
# sample can be obtained from the recorded data
# There are six more images available for reading
# called sample1-6.jpg, feel free to experiment with the others!
image_name = 'sample.jpg'
image = mpimg.imread(image_name)

# Define a function to perform a color threshold
def color_thresh(img, rgb_thresh=(0, 0, 0)):
    img = img.copy()
    img0 = img.copy()
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

    col = Image.fromarray(img)
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<250 else 255, '1')

    return bw
  
  # Define color selection criteria
###### TODO: MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 160
green_threshold = 160
blue_threshold = 160
######
rgb_threshold = (red_threshold, green_threshold, blue_threshold)

# pixels below the thresholds
colorsel = color_thresh(image, rgb_thresh=rgb_threshold)

# Display the original image and binary               
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(colorsel, cmap='gray')
ax2.set_title('Your Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show() # Uncomment if running on your local machine
