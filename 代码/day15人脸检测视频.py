import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#视频检测人脸
camera_number = 0 
cap = cv.VideoCapture(camera_number + cv.CAP_DSHOW)

#在每一帧中检测人脸
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #实例化检测器
        #人脸检测
        face_cas = cv.CascadeClassifier("python_venv_test/pyVenvTest/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        face_cas.load("python_venv_test/pyVenvTest/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        face_rects = face_cas.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors = 11)

        for face_rect in face_rects:
            x,y,w,h = face_rect
            #画出人脸
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

        cv.imshow("face", frame)
        if cv.waitKey(20) & 0xff == ord("q"):
            break

#释放资源
cap.release()
cv.destroyAllWindows()
