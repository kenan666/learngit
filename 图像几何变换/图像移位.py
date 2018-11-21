#  利用API  实现移位    搞清楚算法原理   源代码实现
import cv2
import numpy as np 

img = cv2.imread('1.jpg',1)
cv2.imshow('src',img)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
###
matShift = np.float32([[1,0,100],[0,1,200]])  #  2*3
dst = cv2.warpAffine(img ,matShift,(height,width))  #  1  data ,2  mat, 3 info   #  matShift 偏移矩阵  
#  warpAffine  移位函数
#  移位  矩阵运算
cv2.imshow('dst',dst)
cv2.waitKey(0)

'''
import cv2
import numpy as np

img = cv2.imread('1.jpg',1)
cv2.imshow('src',img)
imgInfo = img.shape
dst = np.zeros (imge.shape,np.uint8)#  uint 8  0-255
height = imgInfo[0]
width = imgInfo[1]
for i in range (0,height ):
    for j in range (0,width-100):
        dst [i,j+100] = img [i ,j]
cv2.imshow('dst',dst)
cv2.waitKey(0)

'''