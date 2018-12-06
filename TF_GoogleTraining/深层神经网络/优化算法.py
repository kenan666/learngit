#  神经网络  优化算法
'''
常用的神经网络  优化算法  --》》  反向传播算法  +  梯度下降法  调整神经网络中  参数的 取值
   1 - 梯度下降法  主要用于优化  单个参数的取值
   2 - 反向传播算法   给出一种高效的  在所有  参数上 使用梯度下降算法  --》 使神经网络模型在训练数据上的  损失函数  尽可能小
   3 - 反向传播算法  是  训练神经网络的核心算法  ，他可以根据  定义好的损失函数  优化  神经网络中参数的 取值，从而使神经网络
       模型在训练数据集上的损失函数达到一个较小值。

***  
  梯度下降法  并不能保证  被优化的函数达到全局最优解
     ---  其中，只有当损失函数为凸函数时，  梯度下降算法才能保证 达到全局最优解。
     
     ---  另一个问题，梯度下降法的  算法运算周期  太长  
          每一轮迭代都要计算所有的  损失函数  计算海量数据的时候，花费的时间太长
     ---  为了 加速训练  ，则 提出  使用随机梯度下降法  来进行  加速运算
        ---  每一轮迭代  只是优化某一条训练数据上的损失函数  虽然加快了运算速度  ，但是，
         ---（缺点)在某一条数据上损失函数更小  并不代表  全部数据上  损失函数更小，  --》》使用随机梯度下降法  甚至  无法达到局部最优

******  解决方案 
每次计算一小部分训练数据的损失函数 ---》这一小部分称为  batch
    --  每次在一个  batch  上优化神经网络的  参数  并不会比  单个数据 慢太多
    --  每次使用   batch    可以大大减少  收敛所需要的  迭代次数   --》》减少了  收敛所需要的  迭代  次数
        同时，可以使收敛结果  更加接近于  梯度下降的结果。

       **   
       batch_size = n

'''

#  神经网络的 进一步优化
'''
通过指数衰减的方法设置   梯度下降法  中的  学习率 ，通过指数衰减的学习率 既可以让模型在训练的前期  快速接近较优解，又可以保证模型在训练
后期不会有太大的波动，从而接近局部最优

指数衰减法  
    tf.train.exponential_decay()  -----  设置参数  stsircase   选择不同的衰减方式   默认值为False
    
    decay_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    
      1 - decay_learning_rate   ---每一轮优化时使用的学习率
      
      2 - learning_rate    --- 事先设定的初始学习率
      
      3 - decay_rate  ---  衰减系数
      
      4 - decay_steps  ---  衰减速度
      
'''

#  过拟合问题
'''
为了避免  过拟合  问题， 通常使用 的 方法是  正则化（regularization）
正则化  --  在损失函数中加入刻画模型复杂程度的指标。

正则化  基本思想是希望通过限制权重的大小，使得模型不能任意拟合训练数据中的随机噪音。

正则化  有两种公式  

其中，L2型正则化的损失函数  定义为：
w = tf.Variable(tf.random_normal([2,1],stddev = 1,seed = 1))
y = tf.matmul (x,w)

loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)
  *  tf.contrib.layers.12_regularizer()函数  可以计算一个给定参数的  L2  正则化的值  

类似  tf.contrib.layers.l1_regularizer()  可以计算  L1  正则化的值

例： 
weights = tf.constant([1.0,-2.0],[-3.0,4.0])
with tf.Session() as sess:
    print (sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))  #  0.5  为正则化权重
    print (sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))

**
神经网络的参数增多的时候  ，  这样的方式可能会  导致  损失函数  定义 很长 ，可读性差且容易出错
更主要的是，当网络结构复杂后，定义网络结构的部分和计算损失函数的部分  可能不在同一个函数内，  这样通过变量这种方式计算损失函数
就不是很方便。

---  解决方案
 tensorflow  中 提供了集合 （collection）
 
 ***注解  collection
          -- 在计算图中，可以通过集合  collection  来管理不同类别的资源
          比如  ： 通过 tf.add_to_collection 函数将资源  加入一个或者多个集合中，通过  tf.get_collection（）获取一个集合中的所有资源
 
'''

#  计算一个  5  层 神经网络  带  L2  正则化的  损失函数  的计算方法
import tensorflow as tf

#  获取一层神经网络边上的权重 ，  并将这个权重 的  L2  正则化 损失  加入  losses  的集合中
def get_weight(shape,lambda ):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    #  add_to_collection  函数 将这个新生成的变量 L2  正则化损失项  加入集合
    #  函数的 第一个参数  losses  是集合的名字，第二个参数 是要加入这个集合的内容
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda)(var))
    # 返回生成的变量
    return var
    
x = tf.placeholder(tf.float32,shape = (None,2))
y_ = tf.placeholder(tf.float32,shape = (None,1))

batch_size = 8

# 定义每一层的节点数
layer_dimension = [2,10,10,10,1]

# 神经网络的层数
n_layers = len(layer_dimension)

# 定义一个变量，这个变量  维护前向传播 时  最深层 的节点  ，开始的时候就是输入层
cur_layer = x

#  当前层的节点数
in_dimension = layer_dimension[0]

# 通过一个循环 来生成  5  层 连接的神经网络结构
for i in range (1,n_layers):
    # layer_dimension[i]  为下一层节点个数
    out_dimension = layer_dimension[i]
    # 生成当前 层中 权重的变量，  并将此变量的  L2  正则损失  加入  计算图上的 集合
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape = [out_dimension]))
    #  使用ReLU 激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
    #  进入下一层之前，将下一层的节点数 更新为当前层的节点数
    in_dimension = layer_dimension[i]
    
#  定义神经网络的 前向传播的同时，已经将 L2  正则化损失加入了 图上的集合
#  现在只需要计算  刻画模型  在训练数据上  表现出来的 损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将  均方差  损失函数  加入损失集合
tf.add_to_collection('losses',mse_loss)

#  get_collection()  返回一个列表，这个列表 是所有这个集合中的元素  ，在这个样例中，
#  这些远  就是损失函数的  不同部分，  将他们加起来就可以得到最终的 损失函数
loss = tf.add_n(tf.get_collection('losses'))
