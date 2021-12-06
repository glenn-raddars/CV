import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

scenery = cv.imread("python_venv_test\pyVenvTest\image\scenery.jpg")
plt.imshow(scenery[:,:,::-1])
plt.show()

rows,cols = scenery.shape[:2]#对shape的元组进行切片（行，列，通道数），切前两个[:2]
print(rows,cols)

#进行图片的缩放
res1 = cv.resize(scenery, (2*cols,2*rows))#等比扩大，用绝对尺寸
plt.imshow(res1[:,:,::-1])
plt.show()

res2 = cv.resize(scenery, None,fx = 0.5,fy = 0.5)#用相对坐标进行缩放（缩小为原来的1/2）
plt.imshow(res2[:,:,::-1])
plt.show()

#图像的平移
M = np.float32([[1,0,100],[0,1,50]])#构建平移矩阵
"""平移矩阵的设置：
[ 1 , 0 , x ]
[ 0 , 1 , y ]第一行是设置在X方向上平移的距离是多少个像素，第二行是设置在y方向上平移多少个像素"""
img1 = cv.warpAffine(scenery, M, (2*cols,2*rows))#warpAffine(图像 ，M平移矩阵 ，显示图像的背景)
plt.imshow(img1[:,:,::-1])
plt.show()

#图像的旋转
"""由x' = rcos(α - θ)，y' = rsin(α - θ)展开推导而成，注意，旋转完后还要换原图像的坐标原点，要将旋转中心平移"""
M = cv.getRotationMatrix2D((cols/2,rows/2), 45, 1)#得到旋转矩阵，（旋转中心，旋转角度，旋转后图像的比例）
img2 = cv.warpAffine(scenery, M, (cols,rows))
plt.imshow(img2[:,:,::-1])
plt.show()

#图像的仿射变换
"""本质上也是一种平移加旋转，https://blog.csdn.net/u011681952/article/details/98942207
有详细解释"""
pts1 = np.float32([[50,35],[100,140],[32,80]])#仿射变换的变换前的3个点
pts2 = np.float32([[50,35],[90,130],[49,56]])#仿射变换后的3个点
M = cv.getAffineTransform(pts1, pts2)#以此求得的仿射变换矩阵

img3 = cv.warpAffine(scenery, M, (cols,rows))#仿射变换
plt.imshow(img3[:,:,::-1])
plt.show()

#图像的透射变换
"""https://blog.csdn.net/oppo62258801/article/details/78642218
透射变换的详细讲解"""

pts1 = np.float32([[12,32],[250,350],[270,105],[390,256]])#透射变换前的四个点
pts2 = np.float32([[24,90],[300,350],[267,267],[390,300]])#透射变换后的四个点
T = cv.getPerspectiveTransform(pts1, pts2)#以此求得透射变换的矩阵,函数为 getPerspectiveTransform

img4 = cv.warpPerspective(scenery, T, (cols,rows))#透射变换的函数也已经发生的变化，为warpPerspective
plt.imshow(img4[:,:,::-1])
plt.show()


#图像金字塔
"""https://blog.csdn.net/zhu_hongji/article/details/81536820
详细讲解了差分金字塔，高斯金字塔以及拉普拉斯金字塔"""
imgUp = cv.pyrUp(scenery)#对图像进行向上采样(实际上是从金字塔顶端朝底端采样，与金字塔方向相反)
plt.imshow(imgUp[:,:,::-1])
plt.show()

imgDown = cv.pyrDown(scenery)#对图像进行向下采样，分辨率和大小减小
plt.imshow(imgDown[:,:,::-1])
plt.show()
