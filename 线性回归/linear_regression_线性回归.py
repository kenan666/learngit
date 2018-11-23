import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

#  x,y  data
x_train = [1,2,3]
y_train = [1,2,3]

# 尝试查找W和b的值来计算y_data = x_data * W + b
# 我们知道W应为1，b应为0
# 但让TensorFlow弄清楚了
W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')
#  求解  XW + b
hypothesis = x_train * W + b

#  定义一个 cost/loss   function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#  创建 session  图
sess = tf.Session()
#  初始化图中的全局变量  global_variables_initializer()
sess.run(tf.global_variables_initializer())

for step in range (2001):
    sess.run (train)
    if step % 20 == 0:
        print (step,sess.run(cost),sess.run(W),sess.run(b))