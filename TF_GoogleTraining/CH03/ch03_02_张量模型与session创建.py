#  张量模型
'''
张量模型  有三个属性  
名字 ： name        维度：shape    类型：type   
'''

import tensorflow as tf

#  tf.constant  是一个计算   这个计算的结果为一个  张量   ，保存在变量  a  中
a = tf.constant ([1.0,2.0],name = 'a')
b = tf.constant([2.0,3.0] ,name = 'b')
result = tf.add(a,b,name ='add')
result = a+b

'''
会话  session

创建一个  sess

sess = tf.Session()

调用是   sess.run(result)
释放资源  sess.close()

创建一个会话，并通过python  中的上下文管理器来管理会话

with tf.Session() as sess:
    #  使用创建好的会话来计算结果
    sess.run()
    #  关闭
    sess.close()
    
上下文  退出时  会话自动关闭和释放资源

*******************
sess = tf.Session()
with sess.as_default():
    print (result_eval())
    
**** 两种不同的方法 ,同样的 结果

sess = tf.Session()
1---
print(sess.run(result))
2---
print (result.eval(session = sess))
*******************

'''