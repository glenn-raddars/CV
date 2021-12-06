import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

blood = cv.imread("python_venv_test\pyVenvTest\image\image.jpg")
plt.imshow(blood[:,:,::-1])
plt.show()

scenery = cv.imread("python_venv_test\pyVenvTest\image\scenery.jpg")
plt.imshow(scenery[:,:,::-1])
plt.show()

#加法(注意，图像大小要求一致)
img1 = cv.add(blood, scenery)#cv中的加法在像素点范围超出255后，会直接默认值为255
plt.imshow(img1[:,:,::-1])
plt.show()

img2 = blood + scenery#而numpy里面的加法是超过255的按照 x % 256 来进行操作，不够好
plt.imshow(img2[:,:,::-1])
plt.show()

#图像混合
img3 = cv.addWeighted(blood, 0.7, scenery, 0.3, 0)#服从的公式是 dst = α*img1 + β*img2 + γ 
plt.imshow(img3[:,:,::-1])
plt.show()
