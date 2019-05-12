#final
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

image = mpimg.imread('grid1.jpg')

shape = image.shape
X = shape[1]
Y = shape [0]

n = 5
factor = .4
def perspect_transform(img, src):
    img = img.copy()
    src1 = src
    (tl, tr, br, bl) = src1

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = (max(int(widthA), int(widthB)))/6
    maxWidth *= factor

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = (max(int(heightA), int(heightB)))/6
    maxHeight *= factor


    dst = np.array([
		[X/2-(maxWidth/2), Y-n ],
		[X/2+(maxWidth/2), Y-n],
		[X/2+(maxWidth/2), Y-n+(maxHeight)],
		[X/2-(maxWidth/2),  Y-n+(maxHeight)]
		], dtype = "float32")
    # Get transform matrix using cv2.getPerspectivTransform()
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp image using cv2.warpPerspective()
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return the result
    return warped

# Define source and destination points
source = np.float32([[117,96], [198,96], [300,141], [14,141]])
#source = np.float32([[73,241], [356,117], [471,260], [190,442]]) scan image
#destination = np.float32([[ , ], [ , ], [ , ], [ , ]])      

warped = perspect_transform(image, source)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show() # Uncomment if running on your local machine



