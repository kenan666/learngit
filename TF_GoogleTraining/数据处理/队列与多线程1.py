import tensorflow as tf

#  创建队列  
q = tf.FIFOQueue(2,"int32")

#  使用enqueue_many 函数来初始化队列中的元素
init = q.enqueue_many(([0,10],))

#  使用Dequeue  函数将队列中的第一个元素出列，，并存储在变量x中
x = q.dequeue()

#  将得到的值 +1
y = x + 1

#  将加 1 后的 值重新加入队列中
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 初始化操作
    init.run()
    for _ in range(5):
        #  运行q_inc 将执行数据出队列 ，出队的 元素 +1  重新加入队列的整个过程
        v, _ = sess.run([x,q_inc])
        print (v)