#   激活函数
'''
激活函数实现去线性化
  ---  激活函数的函数图像都不是一条直线，所以通过激活函数，每一个节点都不再是线性变换，--》神经网路模型 便不再线性
  
常用的激活函数有：
ReLU函数  
sigmoid 函数
tanh 函数
'''
'''
多层网络解决异或问题

利用多层网络解决重要问题  --  异或问题

---   异或问题直观来说  就是  如果输入的两个值的符号相同时  （同为正 或 负）则输出0  ，否则（一个为正一个为负）则输出1
'''

#  损失函数
'''
监督学习  分为两大类   分类 与 回归

提出一个问题：如何判断一个输出向量 和  期望的向量  有多接近呢？  
  --》》 交叉熵 （cross entropy）  是最常用的判断方法之一
  
** 
    交叉熵不是对称的，它刻画的是通过概率分布  q 来表达概率分布 p 的困难程度  ，因为正确答案是希望得到的结果，所以当交叉熵作为神经网络的
损失函数的时候，p 代表正确答案，q 代表预测值

    交叉熵是刻画的两个概率分布的距离，也就是说--》》熵越小，两个概率分布越近
'''
'''
自定义损失函数

经典的损失函数   +  自定义损失函数

例： loss = tf.reduce_sum(tf.where(tf.grater(v1,v2),(v1-v2)*a,(v2-v1)*b))
    --  tf.where  and  tf.grater  实现选择操作4
    tf.grater 的输入是两个张量，此函数比较的是  输入张量的每一个元素的大小，并返回比较结果
    输入的张量维度不一样是，tf会进行类似 numpuy  广播操作的处理。
    tf.where  函数有三个参数：
      第一个  为选择条件 当选择条件为 True  时，tf.where函数会选择  第二个参数的值，否则会选择第三个参数的值4
      * note  tf.where函数判断和选择都是在元素级别进行。
'''
#  自定义损失函数
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# n 输入两个节点
x = tf.placeholder(tf.float32,shape = (None ,2),name = 'x-input')
#  回归问题  一般只有  一个输出  节点
y_ = tf.placeholder(tf.float32,shape = (None ,1),name = 'y-input')

#  定义一个  单层 的神经网络  前向传播的过程  这是  简单的加权和
w1 = tf.Variable(tf.random_normal([2,1],stddev = 1,seed = 1))
y = tf.matmul(x,w1)

#  定义  预测多了  和  预测少了  成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_) * loss_less,(y_ - y) * loss_more))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#  通过 随机数  生成  一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

#  设置 回归的正确值  为  两个输入的和  加上  一个随机量  ，  之所以 要加上一个随机量 是为了加入不可预测的噪音
#   否则  ，不同损失函数  的  意义 就不大了 ， 因为不同  损失函数  都会在  能完全预测正确的时候最低  ，
#  一般来说，噪音为一个均值为 0 的小量  ，所以  这里噪音的设置为  -0.05  ~  0.05  的随机数

Y = [[x1 + x2 +rdm.rand() / 10.0 - 0.05] for (x1,x2) in X]

#  训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run (init_op)
    steps = 5000
    for i in range (steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size , dataset_size)
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        print (sess.run(w1))