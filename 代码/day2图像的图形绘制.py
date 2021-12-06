import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#创建图形背景（画板）
"""返回来一个给定形状和类型的用0填充的数组；
zeros(shape, dtype=float, order=‘C’)
shape:形状
dtype:数据类型，可选参数，默认numpy.float64
order:可选参数，c代表与c语言类似，行优先；F代表列优先"""
img = np.zeros((512,512,3),np.uint8)#大小512*512，通道数3

#绘制图形
cv.line(img, (0,0), (511,511), (255,0,0),5)#直线line(画在哪里(背景)，起始位置，终点位置，颜色(b(蓝),g(绿),r(红)),线条宽度)
cv.circle(img, (256,256), 60, (0,0,255),-1)#圆形cricle(背景，圆心位置，半径，颜色，线条宽度(-1代表填充))
cv.rectangle(img, (100,100), (400,400), (0,255,0),5)#矩形rectangle(背景，左上角位置，右下角位置，颜色，线条宽度)
#写文字
cv.putText(img, "hello", (50,250), cv.FONT_HERSHEY_COMPLEX, 5, (255,255,255),3)
#文字putText(背景，文本，文本框左下角位置，字体，字号大小，颜色，线条宽度，线型)

#显示图像
plt.imshow(img[:,:,::-1])
plt.show()