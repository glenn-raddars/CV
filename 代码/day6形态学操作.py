import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#图像的连通性
"""https://blog.csdn.net/u010622874/article/details/51719556
四连通，八连通，m连通，4邻域，八邻域，D邻域"""



img = cv.imread("python_venv_test\pyVenvTest\image\erzhi.jpg")
plt.imshow(img[:,:,::-1])
plt.show()
"""https://blog.csdn.net/zqx951102/article/details/82997588"""#讲腐蚀与膨胀的
#图像的腐蚀
kernel = np.ones([5,5],np.uint8)#用来做腐蚀的核结构,创建一个值全为一的数组
print(kernel)

img2 = cv.erode(img, kernel)
plt.imshow(img2[:,:,::-1])
plt.show()

#图像的膨胀
img3 = cv.dilate(img, kernel)
plt.imshow(img3[:,:,::-1])
plt.show()

#图像的开闭运算
"""https://blog.csdn.net/afeiererer/article/details/79489150"""#开闭运算以及卷积的讲解
open_img = img
kernel = np.ones([10,10],np.uint8)#核结构
#开运算，先腐蚀，在膨胀，消除周边噪点
opened = cv.morphologyEx(open_img, cv.MORPH_OPEN, kernel)
plt.imshow(opened[:,:,::-1])
plt.show()

#闭运算，先膨胀，在腐蚀，消除结构孔洞
close = cv.imread("python_venv_test\pyVenvTest\image\close.jpg")
closed = cv.morphologyEx(close, cv.MORPH_CLOSE, kernel,None,None,8)#迭代次数超过8次的时候就已经可以完全填充空洞了
plt.imshow(closed[:,:,::-1])
plt.show()

#礼帽运算和黑帽运算
#礼帽运算（顶帽运算）
#原图与开运算结果图之差
#开运算放大了裂缝或者局部低亮度的区域，所以，从原图中减去开运算后的图，
# 得到的结果突出了比原图轮廓周围的区域更明亮的区域，这个操作与选择的核的大小有关。
# TopHat运算一般用来分离比邻近点亮一些的斑块，可以使用这个运算提取背景。
plt.imshow(open_img[:,:,::-1])
plt.show()#原图像
top = cv.morphologyEx(open_img, cv.MORPH_TOPHAT, kernel)
plt.imshow(top[:,:,::-1])
plt.show()

#黑帽运算
#闭运算的结果与原图之差
#黑帽运算的结果突出了比原图轮廓周围区域更暗的区域，所以黑帽运算用来分离比邻近点暗一些的斑块。
plt.imshow(close[:,:,::-1])
plt.show()#原图
black = cv.morphologyEx(close, cv.MORPH_BLACKHAT, kernel,None,None,8)#八次基本可以吧原图中的孔洞完全剥离出来
plt.imshow(black[:,:,::-1])
plt.show()