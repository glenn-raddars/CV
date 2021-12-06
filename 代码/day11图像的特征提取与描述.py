import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Harris角点检测
#角点检测的原理在笔记本上
plt.figure()
blood = cv.imread("python_venv_test\pyVenvTest\image\image.jpg")
plt.subplot(2,2,1)
plt.imshow(blood[:,:,::-1])
#plt.show()

blood_f = np.float32(blood)#Harris角点检测只能作用于格式为float32的图像
plt.subplot(2,2,2)
plt.imshow(blood_f[:,:,::-1])
#plt.show()

blood_g = cv.cvtColor(blood, cv.COLOR_BGR2GRAY)
plt.subplot(2,2,3)
blood_g = np.float32(blood_g)
plt.imshow(blood_g,cmap=plt.cm.gray)
#plt.show()

#角点检测,而且经检验，必须作用于灰度图
dst = cv.cornerHarris(blood_g, 2, 3, 0.04)#原图，角点检测的矩形框大小，sobel算子的卷积核大小，α的取值(0.04,0.06)
blood[dst>0.01*dst.max()] = (0,0,255)#将这个里面所有R数值大于dst中最大值的0.01的焦点全部画出来
plt.subplot(2,2,4)
plt.imshow(blood[:,:,::-1])
plt.show()

#shi_Tomas角点检测
#不需要转换成float32
corners = cv.goodFeaturesToTrack(blood_g, 1000, 0.01, 10)#原图，最多有多少个角点，门限值，角点间的最小距离
print(corners)
for i in corners:
    print("now:",i)
    #专门讲解ravel（）函数的网页，将多维数组转变成一维数组
    x,y = i.ravel()
    """https://blog.csdn.net/liuweiyuxiang/article/details/78220080"""
    cv.circle(blood, (int(x),int(y)), 2, (0,0,255),-1)#画圆的圆心参数不能是小数

plt.imshow(blood[:,:,::-1])
plt.show()