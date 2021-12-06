import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#以默认方式读取图像，彩色，1,0，-1：彩色，灰色，包括alpha通道
img = cv.imread('python_venv_test/pyVenvTest/image/image.jpg',0)

#显示图像
"""cv.imshow('bloodborne', img)"""
#窗口永远等待
"""cv.waitKey(0)"""
"""cv.destroyAllWindows()"""

#matplotlib.pyplot 展示
#首先需要明白一点，我们通过cv2读图片时，数据读取的通道顺序是bgr，并且是height， width， channel的排列方式。
#img[:,:,::-1]也就是我们任意不改变width维的方式，也不改变height维的方式，仅仅改变channel维的方式，
# 并且是倒序排列，原本的bgr排列方式经过倒序就变成了rgb的通道排列方式。
#如果img[::-1, :, :]其实是对图片进行上下翻转， img[:,::-1,:]是对图像进行左右翻转
"""plt.imshow(img[:,:,::-1])"""#在opencv中间是用bgr存储的彩色图像，而在matplotlib里面输出图像是rgb，因此要交换通道
"""plt.show()"""

#以灰度打开
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

#图像的保存
cv.imwrite('python_venv_test\pyVenvTest\image\image.png', img)