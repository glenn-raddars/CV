import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#图像直方图

#以灰度形式打开图像
blood = cv.imread("python_venv_test\pyVenvTest\image\image.jpg",0)
plt.imshow(blood,cmap = plt.cm.gray)
plt.show()

#获取这个图的直方图
hist = cv.calcHist([blood], [0], None, [256], [0,256])#依次参数的意义，图像；哪个通道（这里是灰度图，就一个，用第一个）
#Mask掩膜图像，用来提取图中特殊位置的直方图的，没给就是全图；然后就是直方图的bin（组数）；每个组中的范围；后面两个基本用不到,参数都要用[]
"""https://blog.csdn.net/sunny2038/article/details/9097989
中讲的很详细，可以去看看"""
plt.figure(figsize=(10,8))#创建画布，显示图像，尺寸(10*8)
plt.plot(hist)#https://www.jianshu.com/p/ed3f31fc6a41
#讲解plot的用法，参数等等，这里使用折线图反应直方图
plt.show()


#掩膜作用
#可以提取指定区域的内容
#创建掩膜
mask = np.zeros(blood.shape[:2],np.uint8)#先创建一个跟原图一样大小的全黑蒙板
mask[0:blood.shape[1],314:813] = 1#需要提取的地方置1,第一个参数是区域纵坐标的范围，第二个参数是区域横坐标的范围
plt.imshow(mask,cmap = plt.cm.gray)
plt.show()

mask_img = cv.bitwise_and(blood,blood,mask = mask)#图像按位与，我没搞懂怎么用的，查资料也查不明白，反正要提取掩膜就这么用
plt.imshow(mask_img,cmap = plt.cm.gray)
plt.show()

mask_Hist = cv.calcHist([blood],[0],mask_img,[256],[0,256])#这里图像掩膜无论使用mask还是mask_img结果都一样
plt.plot(mask_Hist)
plt.show()

#直方图均衡化
"""https://blog.csdn.net/schwein_van/article/details/84336633"""
#目的是是图像的对比度更大
dst = cv.equalizeHist(blood)#参数直接输图像
plt.imshow(dst,cmap = plt.cm.gray)
plt.show()

#自适应直方图均衡化
#将图像分成几个小块，每个小块都做直方图均衡化，更能体现细节
cl = cv.createCLAHE(2.0,(8,8))#创建自适应的模板（T）,前一个参数是阈值，后一个是分块的个数
clahe = cl.apply(blood)
plt.imshow(clahe,cmap = plt.cm.gray)
plt.show()