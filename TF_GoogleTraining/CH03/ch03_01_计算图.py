#  计算图  的使用
'''
通过  add计算模型  来计算结果  
add  读取两个常量的取值  
'''
import tensorflow as tf

a = tf.constant([1,2],name = 'a')
b = tf.constant([2,3],name = 'b')
result = a+b

'''
使用  tf.Graph  函数来生成新的计算图  
不同计算图上的张量和运算都不共享
'''

g1 = tf.Graph()
with g1.as_default():
    # 定义一个变量  V  ，并初始化为 0 
    v = tf.get_variable('v',shape = [1],initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    # 定义一个变量  V  ，并初始化为 1
    v = tf.get_variable('v',shape = [1],initializer= tf.ones_initializer)
    
#  在计算图  g1  中读取变量  v  的值
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope(" ",reuse = True):
        #  计算结果为  【0】  ，初始化结果是  0
        print (sess.run(tf.get_variable('v')))
        
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope(" ",reuse = True):
        #  计算结果为  【1】  ，初始化结果是  1
        print (sess.run(tf.get_variable('v')))

'''
使用GPU  计算
'''
#  简单例子  ，后面重点练习
g = tf.Graph()
with g.device('/gpu:0'):
    result = a+ b