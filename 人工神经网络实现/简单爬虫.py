#  爬虫原理
'''
http   html   正则  过滤条件   img url

******************
知识点多：
1   url  
2   html  src
3   img  
4   img  url
'''
import urllib
import urllib3
import os 
from bs4 import BeautifulSoup

#  加载当前url
html = urllib.request.urlopen('https://class.imooc.com/?c=ios&mc_marking=286b51b2a8e40915ea9023c821882e74&mc_channel=L5').read()
#  解析当前 url   1  html   2   html。parser   3  编码  utf-8
soup = BeautifulSoup(html,'html.parser',from_encoding= 'utf-8')
#  img  标签
images = soup.findAll('img')
print (images)
imageName = 0
for image in images:
    link = image.get('src')
    print ('link = ',link)
    link = 'http:'+link
    fileFormat = link[-3:]
    if fileFormat == 'png' or fileFormat == 'jpg':
        fileSavePath = 'C:\\Users\\KeNan\\Desktop\\TL\\刷脸识别\\' + str(imageName) + '.jpg'
        imageName = imageName+1
        urllib.request.urlretrieve(link ,fileSavePath)