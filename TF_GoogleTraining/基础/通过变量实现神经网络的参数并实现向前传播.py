import tensorflow as tf

#  声明  w1  w2  两个变量。通过seed参数设定随机种子
#  这样保证每次结果是一样的
w1 = tf.Variable(tf.random_normal((2,3),stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal((3,1),stddev = 1,seed = 1))

#  暂时将输入的特征向量  定义为一个常量   其中  x  为  1*2 的矩阵
x = tf.constant([[0.7,0.9]])

#  前向传播获取神经网络的输出
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
# 不能用  sess.run(y)直接获取 y 值
#  w1  w2  都没有先初始化   必须先初始化
#     **********  初始化函数  tf.global_variables_initializer()
sess.run(w1.initializer)  #  初始化  w1
sess.run(w2.initializer)  #  初始化  w2

#  输出结果
print(sess.run(y))
sess.close()