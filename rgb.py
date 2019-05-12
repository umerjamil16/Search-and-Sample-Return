from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample.jpg',0)
ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


col = Image.open("sample.jpg")
gray = col.convert('L')
bw = gray.point(lambda x: 0 if x<180 else 255, '1')
plt.imshow(bw)
plt.show()
