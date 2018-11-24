#  完整的训练神经网络的案例  其中步骤可以总结为一下几步：
'''
**************************
1  定义神经网络的结构和前向传播的输出结果
2  定义损失函数  以及选择反向传播优化的算法
3  生成  会话 （session） 并且在训练数据上反复 运行 反向传播优化算法
**************************
这是神经网络 的  基本步骤，，神经网络的结构的变化  ，算法训练步骤  不变
'''
import tensorflow as tf
from numpy.random import RandomState

#  定义训练数据的大小
batch_size = 8

#  定义神经网络的参数  
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
'''
在shape的一个维度  使用 None 可以方便使用不同的  batch 大小  在训练时需要把数据分为较小的
batch，  但是在测试的时候，可以一次性使用全部的数据，   当数据集比较小的时候，这样的比较方便测试  ，
但是数据集比较大的时候，将大量数据放入一个batch  可能会导致内存溢出
'''

#  定义神经网络传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#  定义损失函数和反向传播的算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#  通过随机数生成数据集
rdm = RandomState(1)
dateset_size = 128
X = rdm.rand(dateset_size,2)

'''
定义规则来给出样本的标签  ，在这里所有的  x1 + x2 < 1 的样例都被认为是正样本（比如零件合格）
而其他为负样本 ，（比如零件不合格）。
这里使用  0  和  1 表示  负样本  和正样本  
大部分解决问题的神经网络  都会采用 0  和  1  的表示方法
'''
Y = [[int (x1 + x2 < 1)] for (x1,x2) in X]

#  创建  session 
with tf.Session() as sess:
    init_op = tf.global_variables_initializer ()
    sess.run(init_op)
    
    print (sess.run(w1))
    print (sess.run(w2))
    
    #  设定训练参数 
    steps = 5000
    for i in range (steps):
        #  每次选取  batch_size  个样本训练
        start = (i * batch_size) % dateset_size
        end = min(start + batch_size,dateset_size)
        
        # 通过选取的样本  训练  神经网络  并更新
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            #  每隔一段时间  计算所有数据上的  交叉熵  并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print ('After %d training step(s) ,cross entropy on all data is %g ' %(i ,total_cross_entropy))
            
    print (sess.run(w1))
    print (sess.run(w2))