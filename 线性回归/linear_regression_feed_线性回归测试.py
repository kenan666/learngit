import tensorflow as tf

tf.set_random_seed(777)

# 尝试查找W和b的值来计算y_data = x_data * W + b
# 我们知道W应为1，b应为0
# 但让TensorFlow弄清楚了

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#  现在我们可以用X和Y代替x_data和y_data
##  占位符表示将始终使用feed_dict提供的张量
X = tf.placeholder(tf.float32 ,shape = [None])
Y = tf.placeholder(tf.float32 ,shape = [None])

#  假设 X * W + b
hypothesis = X * W + b

#  定义  cost/loss  函数
cost = tf.reduce_mean (tf.square(hypothesis - Y))

#  minimize  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#  创建    图   在  session  中
sess = tf.Session()

#  初始化全局变量  
sess.run(tf.global_variables_initializer())

#  线性拟合
for step in range (2001):
    cost_val ,W_val ,b_val ,_ = sess.run([cost,W,b,train],feed_dict = {X:[1,2,3],Y:[1,2,3]})