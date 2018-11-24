import tensorflow as tf

#  模型参数
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

# 输入  输出
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = x * W + b  

#  定义  cost/ loss 函数
loss = tf.reduce_sum (tf.square(linear_model - y))

#  optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)  #  只适应学习率
train = optimizer.minimize(loss)

#  训练数据
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# 循环训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})
    
#  评估准确性
curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))