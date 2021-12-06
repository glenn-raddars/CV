import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
#模板匹配
blood = cv.imread("python_venv_test\pyVenvTest\image\image.jpg")
plt.subplot(2,2,1)
plt.imshow(blood[:,:,::-1])


head = cv.imread("python_venv_test\pyVenvTest\image\head.jpg")
plt.subplot(2,2,2)
plt.imshow(head[:,:,::-1])


#start
res = cv.matchTemplate(blood, head, cv.TM_CCORR_NORMED)#三个参数原图，模板，匹配方式（参考以下网址）
"""https://www.cnblogs.com/xrwang/archive/2010/02/05/MatchTemplate.html"""
#将计算结果矩阵以图像形式输出
plt.subplot(2,2,3)
plt.imshow(res,cmap = plt.cm.gray)

#用minmaxloc函数找到输出矩阵中的最大值（就是匹配矩形框的左上角）
min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
#定位左上角
top_left = max_loc
print(top_left)
#拿到模板大小
h,w = head.shape[:2]
#定位右下角
bottem_right = (top_left[0]+w, top_left[1]+h)
print(bottem_right)
#在原图匹配位置画矩形框
cv.rectangle(blood, top_left, bottem_right, (0,255,0),5)
plt.subplot(2,2,4)
plt.imshow(blood[:,:,::-1])
plt.show()

#霍夫线检测(霍夫变换)
detect_line = cv.imread("python_venv_test\pyVenvTest\image\huofuLine.jpg")
plt.figure()
plt.subplot(2,2,1)
plt.imshow(detect_line[:,:,::-1])

#canny边缘检测
edg = cv.Canny(detect_line, 50, 100)
plt.subplot(2,2,2)
plt.imshow(edg,cmap=plt.cm.gray)

#在边缘提取的基础上检测直线
lines = cv.HoughLines(edg, 0.8, np.pi/180, 130)#参数，原图，rho的单位精度，θ的单位精度，阈值
#绘制直线
for line in lines:
    print(line)
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho*a
    y0 = rho*b
    #再找两个点，绘制直线，使其尽量沾满整个图像
    """https://www.it610.com/article/1291807161583738880.htm
    详细讲解，为何乘以1000"""
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)
    cv.line(detect_line, (x1,y1), (x2,y2), (0,255,0))
#可以见得检测效果并不好
plt.subplot(2,2,3)
plt.imshow(detect_line[:,:,::-1])

#用HoughlineP来检测线段,说实话效果也一般
#https://blog.csdn.net/czcty/article/details/97658235
lines = cv.HoughLinesP(edg, 1.0, np.pi/180, 70,maxLineGap = 20)
for line in lines:
    print(line)
    x1,y1,x2,y2 = line[0]
    cv.line(detect_line, (x1,y1), (x2,y2), (0,255,0))

plt.subplot(2,2,4)
plt.imshow(detect_line[:,:,::-1])
plt.show()

#霍夫圆检测
"""https://blog.csdn.net/qq_41498261/article/details/103104035"""
cricle = cv.imread("python_venv_test\pyVenvTest\image\cricle.jpg")
plt.figure()
plt.subplot(2,2,1)
plt.imshow(cricle[:,:,::-1])
#plt.show()
#转灰度图
cricle_g = cv.cvtColor(cricle, cv.COLOR_BGR2GRAY)
plt.subplot(2,2,2)
plt.imshow(cricle_g,cmap=plt.cm.gray)
#用中值滤波去噪，因为霍夫圆检测对噪声比较敏感
cricle_s = cv.medianBlur(cricle_g, 7)
plt.subplot(2,2,3)
plt.imshow(cricle_s,cmap=plt.cm.gray)
#plt.show()

#开始霍夫圆检测,只能做灰度图
cricle_d = cv.HoughCircles(cricle_s, cv.HOUGH_GRADIENT, 1,200,param1 = 100,param2 = 50,minRadius = 0,maxRadius = 200)
#几个参数在所给的网站中都有详细的解释，这里不再赘述
print(cricle_d)

#绘制圆
for i in cricle_d[0]:
    print(i)
    cv.circle(cricle, (int(i[0]),int(i[1])), int(i[2]), (0,255,0),3)#圆
    cv.circle(cricle, (int(i[0]),int(i[1])), 2, (0,0,255),-1)#圆心

plt.subplot(2,2,4)
plt.imshow(cricle[:,:,::-1])
plt.show()