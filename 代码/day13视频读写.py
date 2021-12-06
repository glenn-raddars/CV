import cv2 as cv
import numpy as np

#视频读取
cap = cv.VideoCapture('python_venv_test/pyVenvTest/video/VID_20210125_131818.mp4')#硬是打不开，打算跳过
#后来又尼玛的打开了，非要用\\或者/
#判断视频是否读取成功
x = cap.isOpened()
print(x)
while(x):
    #获取每一帧图像
    ret,frame = cap.read()#返回图像是否读取成功的布尔值，和每一帧的图像
    print(ret)
    
    #如果获取成功
    if ret == True:
        cv.imshow("frame", frame)
    #在运行25ms或者按下q后结束
    if cv.waitKey(25) & 0xff == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

#图像输出
cap = cv.VideoCapture('python_venv_test/pyVenvTest/video/VID_20210125_131818.mp4')#打开视频
width = int(cap.get(3))#获取视频宽度
height = int(cap.get(4))#获取视频高度
#输出地址，输出视频格式，帧数，输出视频大小
out = cv.VideoWriter("python_venv_test/pyVenvTest/video/friend.avi",cv.VideoWriter_fourcc("M", "J", "P", "G"),60,(width,height))

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        out.write(frame)#将每一帧图片输出
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()

