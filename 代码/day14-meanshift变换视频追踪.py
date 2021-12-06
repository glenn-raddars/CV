import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
cap = cv.VideoCapture("python_venv_test/pyVenvTest/video/VID_20210125_131818.mp4")
ret,frame = cap.read()

plt.imshow(frame[:,:,::-1])
plt.show()
r,h,c,w = 0,330,369,392#要追踪位置的行高列宽，行和高对应纵坐标，列和宽对应横坐标
cv.rectangle(frame, (369,0), ((369+392),(0+330)), (0,0,255),2)
plt.imshow(frame[:,:,::-1])
plt.show()

win = (c,r,w,h)#追踪窗口，列行宽高
roi = frame[r:r+h,c:c+w]#先高在宽，先纵坐标，在横坐标
plt.imshow(roi[:,:,::-1])
plt.show()

#计算直方图
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
roi_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])#hsv图像中的h亮度这个通道，范围就到180
cv.normalize(roi_hist, roi_hist,0,255,cv.NORM_MINMAX)#归一化函数
#输入图像，输出图像，归一化最小值，归一化最大值，归一化类型这里主要是为了把0到180的空间映射到0到255，
# 去跟视频中每一帧图像进行比较
"""https://blog.csdn.net/lanmeng_smile/article/details/49903865"""
#讲归一化函数的

#目标追踪
term = (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,1)#设置终止条件，最大迭代次数，中心飘离最小次数

while(True):
    ret,frame = cap.read()
    if ret == True:
        hst = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hst],[0] , roi_hist, [0,180], 1)#返向直方图，原图，通道，模板直方图，范围，组距

        ret,win = cv.meanShift(dst, win, term)#在哪里做meanshift，窗口，终止条件
        x,y,w,h = win
        img = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),1)
        cv.imshow("frame", img)
        if cv.waitKey(60) & 0xff == ord("q"):
            break

cap.release()

