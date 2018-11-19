#  opencv tensorflow
#  类比  语法  api  原理
#  基础数据类型  运算符   流程  字典 数组

import tensorflow as tf
data1 = tf.constant(2.5)
data2 = tf.Variable(10,name = 'var')
print(data1)
print (data2)
'''
sess = tf.Session()
print (sess.run(data1))
init = tf.global_variables_initializer()
sess.run(init)
print (sess.run(data2))
sess.close()
'''
init = tf.global_variables_initializer()
sess = tf.Session()
with sess:
    sess.run(init)
    print (sess.run(data2))


#  TensorFlow  本质
#  tf = tensor + 计算图
#  op  operation
#  graphs  数据操作
#  session