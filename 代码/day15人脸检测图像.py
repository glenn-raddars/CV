import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

face = cv.imread("python_venv_test/pyVenvTest/image/face1.jpg")
face_g = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
plt.imshow(face_g,cmap = plt.cm.gray)
plt.show()

#实例化检测器
#人脸检测
face_cas = cv.CascadeClassifier("python_venv_test/pyVenvTest/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
face_cas.load("python_venv_test/pyVenvTest/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
#眼睛的检测
eyes_cas = cv.CascadeClassifier("python_venv_test\pyVenvTest\Lib\site-packages\cv2\data\haarcascade_eye.xml")
eyes_cas.load("python_venv_test\pyVenvTest\Lib\site-packages\cv2\data\haarcascade_eye.xml")

"""https://blog.csdn.net/leaf_zizi/article/details/107637433"""
#人脸检测函数detectMutiscale的讲解
face_rects = face_cas.detectMultiScale(face_g,scaleFactor = 1.1,minNeighbors = 3,minSize = (32,32))#原图，图像缩小比例系数，最小相邻框
#最小监测尺寸
#绘制人脸框
for face_rect in face_rects:
    x,y,w,h = face_rect
    cv.rectangle(face, (x,y), (x+w,y+h), (0,255,0),2)
    #接下来画眼睛
    eye_color = face[y:y+h,x:x+w]#眼睛所在的区域，彩图
    eye_gray = face_g[y:y+h,x:x+w]#眼睛所在的区域，灰度图
    eyes_rects = eyes_cas.detectMultiScale(eye_gray,scaleFactor = 1.3,minNeighbors = 6)
    for eyes_rect in eyes_rects:
        ex,ey,ew,eh = eyes_rect
        cv.rectangle(eye_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)

plt.imshow(face[:,:,::-1])
plt.show()


face = cv.imread("python_venv_test/pyVenvTest/image/face2.jpg")
face_g = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
plt.imshow(face_g,cmap = plt.cm.gray)
plt.show()

#实例化检测器
#人脸检测
face_cas = cv.CascadeClassifier("python_venv_test/pyVenvTest/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
face_cas.load("python_venv_test/pyVenvTest/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
#眼睛的检测
eyes_cas = cv.CascadeClassifier("python_venv_test\pyVenvTest\Lib\site-packages\cv2\data\haarcascade_eye.xml")
eyes_cas.load("python_venv_test\pyVenvTest\Lib\site-packages\cv2\data\haarcascade_eye.xml")

"""https://blog.csdn.net/leaf_zizi/article/details/107637433"""
#人脸检测函数detectMutiscale的讲解
face_rects = face_cas.detectMultiScale(face_g,scaleFactor = 1.017,minNeighbors = 20)#原图，图像缩小比例系数，最小相邻框
#最小监测尺寸
#绘制人脸框
for face_rect in face_rects:
    x,y,w,h = face_rect
    cv.rectangle(face, (x,y), (x+w,y+h), (0,255,0),2)
    #接下来画眼睛
    #eye_color = face[y:y+h,x:x+w]#眼睛所在的区域，彩图
   # eye_gray = face_g[y:y+h,x:x+w]#眼睛所在的区域，灰度图
    #eyes_rects = eyes_cas.detectMultiScale(eye_gray,scaleFactor = 1.3,minNeighbors = 6)
    #for eyes_rect in eyes_rects:
        #ex,ey,ew,eh = eyes_rect
       # cv.rectangle(eye_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)

plt.imshow(face[:,:,::-1])
plt.show()


