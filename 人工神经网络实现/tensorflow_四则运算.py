import tensorflow as tf

data1 = tf.constant(6)
data2 = tf.constant(2)
dataAdd = tf.add(data1,data2)
dataMul = tf.multiply(data1,data2)
dataSub = tf.subtract(data1,data2)
dataDiv = tf.divide(data1,data2)
with tf.Session() as sess:
    print (sess.run(dataAdd))
    print (sess.run(dataMul))
    print (sess.run(dataSub))
    print (sess.run(dataDiv))
print('end!')
'''
import tensorflow as tf

data1 = tf.constant(6)
data2 = tf.Variable(2)
dataAdd = tf.add(data1,data2)
dataCopy = tf.assign(data2,dataAdd)#  dataAdd->data2
dataMul = tf.multiply(data1,data2)
dataSub = tf.subtract(data1,data2)
dataDiv = tf.divide(data1,data2)
init = tf.global_variables_initializer()#  变量必须初始化，才能进行运算
with tf.Session() as sess:
    print (sess.run(init))
    print (sess.run(dataAdd))
    print (sess.run(dataMul))
    print (sess.run(dataSub))
    print (sess.run(dataDiv))
    print ('sess.run(dataCopy)',sess.run(dataCopy))  # 8  ->data2
    print ('dataCopy.eval()',dataCopy.eval())# 8+6->14
    print ('tf.get_default_session()',tf.get_default_session().run(dataCopy))
print('end!')

'''