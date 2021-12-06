import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
blood = cv.imread("python_venv_test\pyVenvTest\image\image.jpg")
plt.subplot(2,2,1)
plt.imshow(blood[:,:,::-1])


blood_g = cv.cvtColor(blood, cv.COLOR_BGR2GRAY)
plt.subplot(2,2,2)
plt.imshow(blood_g,cmap = plt.cm.gray)


sift = cv.xfeatures2d.SIFT_create()#创建一个sift对象
"""https://blog.csdn.net/qq_36387683/article/details/80559237"""#讲这两个函数的
kp,des = sift.detectAndCompute(blood_g,None)#返回关键点和其描述符,后面一个参数是掩膜
cv.drawKeypoints(blood,kp, blood,flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#绘制关键点，输入图像，关键点，输出图像，关键点的表现形式
plt.subplot(2,2,3)
plt.imshow(blood[:,:,::-1])
plt.show()

#fast角点检测
"""https://blog.csdn.net/lyl771857509/article/details/79661483"""
plt.figure()
fast = cv.FastFeatureDetector_create(threshold = 30,nonmaxSuppression = 1)#threshold=None(阈值), nonmaxSuppression=None（是否进行非最大化抑制）
kp2 = fast.detect(blood,None)#没有特征点的描述，只有检测,可以用彩图
img2 = cv.drawKeypoints(blood, kp2, None,(0,0,255))
plt.subplot(1,2,1)
plt.imshow(img2[:,:,::-1])

#不进行非最大化抑制,更加密集
fast.setNonmaxSuppression(0)
kp3 = fast.detect(blood,None)
img3 = cv.drawKeypoints(blood, kp3, None,(0,0,255))
plt.subplot(1,2,2)
plt.imshow(img3[:,:,::-1])
plt.show()

#orb算法
"""https://blog.csdn.net/yang843061497/article/details/38553765"""
plt.figure()
img = cv.imread("python_venv_test\pyVenvTest\image\image.png")
plt.subplot(2,2,1)
plt.imshow(img[:,:,::-1])

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(2,2,2)
plt.imshow(gray,cmap = plt.cm.gray)

orb = cv.ORB_create(nfeatures = 5000)#表示采取5000个特征点
kp,des = orb.detectAndCompute(img,None)#可以用彩图
img2 = cv.drawKeypoints(img, kp, None,flags = 0)#画特征点
plt.subplot(2,2,3)
plt.imshow(img2[:,:,::-1])
plt.show()