import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#先绘制一个全黑的图像
img = np.zeros((256,256,3),np.uint8)
#显示图像
plt.imshow(img[:,:,::-1])
plt.show()

#提取像素
print(img[100,100])
#只返回单通道的值
print(img[100,100,0])
#修改像素点的值
img[100,100] = (255,255,3)
plt.imshow(img[:,:,::-1])
plt.show()

#再次查看像素点的单通道值
print(img[100,100,0])
print(img[100,100,1])
print(img[100,100,2])

#图像的属性展示(都是以数组形式展现)(别带括号)
print(img.shape)#形状
print(img.dtype)#类型
print(img.size)#大小

#拆分与合并图像的通道
blood = cv.imread("python_venv_test\pyVenvTest\image\image.jpg",-1)
plt.imshow(blood[:,:,::-1])
plt.show()

b,g,r = cv.split(blood)
plt.imshow(b,cmap=plt.cm.gray)
plt.show()

plt.imshow(g)
plt.show()

plt.imshow(r)
plt.show()
#合并
img2 = cv.merge((b,g,r))
plt.imshow(img2[:,:,::-1])
plt.show()

#转换图片格式
gray = cv.cvtColor(blood, cv.COLOR_BGR2GRAY)#cv.COLOR_BGR2GRAY是将彩图转化为灰度图
plt.imshow(gray,cmap=plt.cm.gray)
plt.show()

hsv = cv.cvtColor(blood, cv.COLOR_BGR2HSV)#cv.COLOR_BGR2HSV是将彩图转化为HSV格式图
plt.imshow(hsv)
plt.show()