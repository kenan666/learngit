#  完成图片的加载  ->获取图片的 信息 -> 图片缩放  ->检查
import cv2

img = cv2.imread('1.jpg',1)
imgInfo = img.shape
print (imgInfo)
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
# 1 放大  2  缩小  3  等比例
dstHeight = int (height * 0.5)
dstWidth = int (width * 0.5)
# 最近邻域差值  双线性差值  像素关系重采样  立方差值
dst = cv2.resize(img ,(dstWidth,dstHeight))   #  使用resize  方法   API  resize
cv2.imshow('image',dst)
cv2.waitKey(0)

'''
import cv2 
import numpy as np

#  原理
'''
# [ [A1 A2 B1],[A3 A4 B2]]
# [ [A1 A2],[A3 A4]]  [ [B1] ,[B2]]
# newX = A1 *x + A2 *y + B1
# newY = A3 * x + A4 * y +B2
# x->x*0.5  y->y*0.5
# newX = 0.5*x
'''

img = cv2.imread('1.jpg',1)
cv2.imshow('src',img)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
matScale = np.float32([[0.5,0,0],[0,0.5,0]])
dst = cv2.warpAffine(img,matScale,(int (width /2),int (height /2)))
cv2.imshow('dst',dst)
cv2.waitKey(0)

'''

#  方法3
'''
#  获取信息   空白模板   xy坐标
import cv2
import numpy as np

#  图片缩放
img = cv2.imread('1.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
dstHeight = int (height / 2)
dstWidth = int (width / 2)
#  最近邻域插值法
dstImage = np.zeros((dstHeight,dstWidth,3),np.uint8)
for i in range(0,dstHeight):  # 行
    for j in range (0,dstWidth):  #  列
        iNew = int(i*(height*1.0/dstHeight))
        jNew = int (j*(width*1.0/dstWidth))
        dstImage [i,j] = img [iNew,jNew]
cv2.imshow('dst',dstImage)
cv2.waitKey(0)

'''