# 从Opencv中导入函数

import cv2 as cv


#创建一个窗口，命名为you need tostruggle,
#cv.CV_WINDOW_AUTOSIZE 这个参数设定显示窗口虽图片大小自动变化
cv.namedWindow('You need to struggle', cv.WINDOW_AUTOSIZE)

#加载一张图片，第二个参数指定当图片被加载后的格式，还有另外两个可选参数
#CV_LOAD_IMAGE_GREYSCALE and CV_LOAD_IMAGE_UNCHANGED，分别是灰度格式和不变格式
imagePath = '0-common_pics/common_1.jpg'
image = cv.imread(imagePath)

#创建一个矩形，来让我们在图片上写文字，参数依次定义了文字类型，高，宽，字体厚度等。。


#将文字框加入到图片中，(5,20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色
fontface = cv.FONT_HERSHEY_SCRIPT_SIMPLEX  # 使用默认字体
'''
rectangle绘制矩形(参数 img  起点坐标  宽高  颜色  thickness线宽)
'''
cv.rectangle(image, (10, 10), (110, 110), (0, 0, 255), thickness=2)  # cv2.floodFill()
cv.rectangle(image, (12, 12), (108, 38), (255, 0, 0), thickness=-1 )
cv.putText(image, "Hello World你好", (30, 30), fontface,0.3, (0, 255, 0))

#在刚才创建的窗口中显示图片
cv.imshow(u'You need to struggle你好', image)
cv.waitKey(0)

#保存图片
cv.imwrite('E://feel.png', image)