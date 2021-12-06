import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

salt_and_pepper = cv.imread("python_venv_test\pyVenvTest\image\salt_and_pepper.jpg")#椒盐噪声的图像
gauss = cv.imread("python_venv_test\pyVenvTest\image\goss.jpg")#高斯噪声的图像
plt.imshow(salt_and_pepper[:,:,::-1])
plt.show()
plt.imshow(gauss[:,:,::-1])
plt.show()

#均值滤波
s_and_p_did = cv.blur(salt_and_pepper, (5,5))
for num in range(1,10):#做一次不够明显，多做几次
    s_and_p_did = cv.blur(s_and_p_did, (5,5))#blur(src, ksize, dst=None, anchor=None, borderType=None, /)
#参数依次是原图，卷积核大小，dst是目标图像大小，一般不管，anchor是开始卷积的位置，默认为（-1，-1）及核的中心
#最后是边界类型，一般不管
plt.imshow(s_and_p_did[:,:,::-1])
plt.show()

#用均值滤波率高斯噪声,效果不明显
gaussed = cv.blur(gauss, (5,5))
for num in range(1,5):
    gaussed = cv.blur(gaussed, (5,5))
plt.imshow(gaussed[:,:,::-1])
plt.show()

#高斯滤波
"""https://blog.csdn.net/keith_bb/article/details/54412493
讲高斯滤波函数Gaussianblur的"""

#用高斯滤波率高斯噪声
aftergauss = cv.GaussianBlur(gauss, (3,3), 1)
for i in range(1,10):#他也要滤很多次才能看出来区别，很奇怪,而且改变滤波核的大小以及水平，垂直方向上的方差没有什么区别
    aftergauss = cv.GaussianBlur(aftergauss, (3,3), 1)
plt.imshow(aftergauss[:,:,::-1])
plt.show()

#用中值滤波去除椒盐噪声
aftermed = cv.medianBlur(salt_and_pepper, 9)#参数，图像，滤波核大小（只是一个整数）,核越大越模糊
plt.imshow(aftermed[:,:,::-1])
plt.show()