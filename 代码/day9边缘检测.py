import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#边缘检测
#无非就是一阶导的最大值和二阶导数的零点

#Sobel算子，一阶导数检测
#cv.Sobel(src, ddepth, dx, dy,)
# Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None)
#src做边缘检测的图像，ddepth图像的深度（用来扩张通道大小，上限不再是255），dx横方向上的，
# dy纵方向上的，ksize卷积核大小(如果为-1则转换成Scharr算子)，scale缩放导数的比例系数默认没有，
# borderType边界类型，不用管
#由于扩张了通道大小，最后要用cv.convertScaleAbs(src)转换为原来的unit8格式
#使用figure和subplot在同一张画布上显示图像
plt.figure()#创建一张画布
"""https://www.cnblogs.com/The-Shining/p/11700093.html
专门讲figure和subplot的用法的"""
blood = cv.imread("C:\python\python_venv_test\pyVenvTest\image\image.jpg",-1)
plt.subplot(2,3,1)#分区
plt.imshow(blood[:,:,::-1])
#plt.show()

#开始边缘检测
x = cv.Sobel(blood, cv.CV_64F, 1, 0)#x方向,要整一个能有负数的深度，一般选择CV_64F
y = cv.Sobel(blood, cv.CV_64F, 0, 1)#y方向
#将两个方向都转化成unit8格式
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
#最后合并
res = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
plt.subplot(2,3,2)
plt.imshow(res[:,:,::-1])
#plt.show()

#Scharr算子,显示的细节更多，但很冗余，有的时候不需要
x = cv.Sobel(blood, cv.CV_64F, 1, 0,ksize = -1)#x方向,要整一个能有负数的深度，一般选择CV_64F
y = cv.Sobel(blood, cv.CV_64F, 0, 1,ksize = -1)#y方向
#将两个方向都转化成unit8格式
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
 
 #最后合并
res = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
plt.subplot(2,3,3)
plt.imshow(res[:,:,::-1])
#plt.show()

#laplacian算子求边缘
img = cv.Laplacian(blood, cv.CV_16S)#三个参数,原图，深度（同上），ksize卷积核大小，只能为奇数
res = cv.convertScaleAbs(img)
plt.subplot(2,3,4)
plt.imshow(res[:,:,::-1])
#plt.show()

#canny边缘检测
img = cv.Canny(blood, 100, 200)#参数，原图，两个门限值，低门限值，高门限值
plt.subplot(2,3,5)
plt.imshow(img,cmap = plt.cm.gray)

temp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = cv.add(blood, temp)
plt.subplot(2,3,6)
plt.imshow(img2[:,:,::-1])
plt.show()

plt.imshow(temp[:,:,::-1])
plt.show()