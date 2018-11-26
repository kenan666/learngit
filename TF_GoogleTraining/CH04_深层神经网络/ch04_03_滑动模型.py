#  滑动模型
'''
滑动模型 --- 可以使  模型在 数据集上更健壮的方法
在采用随机梯度下降算法  训练神经网络  时 ， 使用滑动平均模型在很多应用中  都可以在一定程度上提高  最终模型 在测试数据集上的表现

tf.ExponentialMovingAverage()  + decay()(衰减率,控制模型更新的速度)
通过上述变化  得到  每一个一一对相应的影子变量（shadow_variable）:
        这个 影子变量的 初始值 就是相对应  变量的 初始值  ，而每次运行变量更新时，，应自变量的值 会更新为：
        shadow_variable = decay * shadow_variable + (1 - decay) * variable
        ---    shadow_variable  为影子变量  ，variable  为待更新变量  decay  为衰减率
        *  decay  决定了模型更新的速度 ，  decay  越大模型越稳定  实际应用中  ，decay  一般设置为接近 1 的数 ，（0.99,0.999）
        或者  利用 num_updates（动态控制  decay 大小）

'''

#  ExponentialMovingAverage 的 使用
import tensorflow as tf

#  定义一个变量用于计算  滑动平均  这个变量的初始值为 0 ，---此时手动指定的滑动变量的大小
# 类型为 tf.float32  滑动变量平均  必须是  实数型
v1 = tf.Variable(0,dtype = tf.float32)
# 定义 step  变量， 模拟神经网络  中迭代  的  轮数，可以用于动态控制  衰减率
step = tf.Variable(0,trainable=False)

# 定义一个滑动平均类 （class）  ，初始化 时  给定衰减率（0.99）  和控制衰减率的变量  step
ema = tf.train.ExponentialMovingAverage(0.99,step)  #  给定参数值

#  定义一个  更新变量滑动平均的操作，这里需要给定  一个列表，  每次执行这个操作时，这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    # 通过 ema.average  (v1)  获取滑动平均 之后  变量的值  。  在初始化  之后变量  v1  的值 和  v1  的滑动平均  都为  0
    print(sess.run([v1,ema.average(v1)]))  # 此时 输出  【0,0】
    
    # 更新 变量  v1  的值 到  5
    sess.run(tf.assign(v1,5))
    #  更新  v1 的滑动平均值。  衰减率为min{0.99，（1 + step）/ （10 + step） = 0.1} = 0.1
    #  所以此时  v1 的滑动平均  会被更新为  0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print (sess.run([v1,ema.average(v1)]))  # 此时  输出 【5,4.5】
    
    #  更新step 为 10000
    sess.run(tf.assign(step,10000))
    #  更新  v1  的  值为  10
    sess.run(tf.assign(v1,10))
    
    #  更新v1 的滑动平均值 ，衰减率  为min{0.99，（1+step）/（10+step） = 0.99} = 0.99
    #  所以v1的滑动平均  会被更新为  0.99*4.5+0.01*10 = 4.555
    sess.run(maintain_averages_op)
    print (sess.run([v1,ema.average(v1)]))  #  此时  输出  【10,4.555】
    
    #  再次  更新平均滑动  ，得到的新滑动平均值  为  0.99*4.555+0.01*10=4.60945
    sess.run(maintain_averages_op)
    print (sess.run([v1,ema.average(v1)]))  #  此时  输出  【10,4.6094499】  
    