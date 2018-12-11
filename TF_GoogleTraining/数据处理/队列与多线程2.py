'''
tf  中提供了两种类来完成多线程协同的功能
tf.Coordinator()---主要用于协同多个进程一起停止并提供  should_stop   request_stop  join 三个函数
1、should_stop 函数，当函数返回值为true时，则当前线程  也需要退出
2、request_stop每调用一个都可以使用 这个函数通知其他线程退出---当调用request_stop 之后，should_stop 返回值会被设置为true，其他线程终止
    
tf.QueueRunner()
'''
#  tf.Coordinator()  函数方法的使用

import tensorflow as tf
import numpy as np
import threading
import time

def MyLoop(coord,worker_id):
    #  使用tf.Coordinator  类提供的协同工具判断当前线程是否停止
    while not coord.should_stop():
        #  随机停止所有线程
        if np.random.rand() < 0.1:
            print ("stoping from id :%d\n" % worker_id)
            
            # 调用  coord。request_stop（）函数来通知其他线程停止
            coord.request_stop()
        else:
            #  打印当前线程  id
            print ("working on id : %d\n" % worker_id)
        
        # 暂停  1秒
        time.sleep(1)
        
#  声明一个  tf.train.Coordinator 类  来协同多个线程
coord = tf.train.Coordinator()

#  创建 5  个线程
threads = [threading.Thread(target = MyLoop,args = (coord,i,)) for i in range(5)]

#  启动所有线程
for t in threads :
    t.start()

#  等待所有线程退出
coord.join(threads)


# tf.QueueRunner函数的使用
#  声明一个先进先出的队列  ，队列中最多有100  个元素
queue = tf.FIFOQueue(100,"float")

#  定义入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])
 
#  使用 tf,train.QueueRunner  来创建多个线程队列的入队操作
#  tf,train.QueueRunner的第一个参数给出了被操作的队列  
#  【enqueue_op】 * 5  表示需要启动 5  个线程  每个线程都是enqueue_op 操作
qr = tf.train.QueueRunner(queue,[enqueue_op] * 5)

# 将定义的QueueRunner  加入TensorFlow  计算图上指定集合
#tf.train.add_queue_runner  函数没有指定集合，则加入默认集合 tf.GraphKeys.QUEUE_RUNNERS 
tf.train.add_queue_runner(qr)

#  定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用tr.train.Coordinator 来协同启动进程
    coord = tf.train.Coordinator()
    
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    # 获取队列中的值
    for _ in range (3):
        print (sess.run(out_tensor)[0])
    
    #  使用tf.train.Coordinator  来停止所有进程
    coord.request_stop()
    coord.join(threads)