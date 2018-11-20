#  步骤  -》 load  xml  文件   -》   load  图片    -》灰度处理  -》  检测  ->  遍历并标注
import cv2
import numpy as np

#  load  xml  file 
face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
#  load  jpg  file 
img =  cv2.imread('face.jpg')
cv2.imshow('src',img)
# 计算  haar  特征
gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#  检测
faces = face_xml.detectMultiScale(gray,1.3,5)
print ('face =',len (faces))
#  draw  
index = 0
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #  rectangle  绘制方框  标记
    roi_face = gray[y:y+h,x:x+w]
    roi_color = img [y:y+h,x:x+w]
    fileName = str(index) + '.jpg'
    cv2.imwrite(fileName,roi_color)
    index = index + 1
    #  必须是灰度图像
    eyes = eye_xml.detectMultiScale(roi_face)
    print ('eye = ',len(eyes))
    #for (e_x,e_y,e_w,e_h) in eyes:
        #cv2.rectangle(roi_color,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)    
cv2.imshow('dst',img)
cv2.waitKey(0)