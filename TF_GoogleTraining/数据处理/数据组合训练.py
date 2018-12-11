'''
将多个输入样例组织成 一个 batch  可以提高模型训练的效率
单个样例的预处理结果后，将他们组织成 batch 然后提供给神经网络的输入层，

tf  中提供了  tf.train.batch  and  tf.train.shuffle_batch  函数  将单个样例  组织成batch的形式输出

两个函数都会生成  一个队列  ，队列的入队操作 是  生成单个样例的方法，而每次出队得到的是一个batch  的样例

'''

import tensorflow as tf

#  tf.train.batch  函数用法

example, label = features['i'], features['j']
batch_size = 3

#  组合样例中最大存储的样例个数
capacity = 1000 + 3 * batch_size

#  使用tf.train.batch  函数来组合样例
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #  获取并打印组合后的样例
    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print (cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)


#  tf.train.shuffle_batch  函数用法
'''
example, label = features['i'], features['j']

example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue = 30)

其余代码  基本一样
'''
'''
tf.train.shuffle_batch  函数中通过设置参数  num_threads 的值  ，指定多个线程同时执行入队操作，
---》其入队操作就是数据读取以及预处理的过程

当  num_threads 的值 大于 1  时，多个线程会同时读取一个文件中的不同样例并进行预处理，
如果需要多个线程处理不同的文件中的样例，可以使用 tf.train.shuffle_batch_join 函数  --》从输入队列中获取不同的文件分配给不同的线程


tf.train.shuffle_batch不同线程会读取同一个文件   ，如果文件中的样例比较类似，可能会受到影响   ---》所以需要尽量打乱文件
tf.train.shuffle_batch_join 不同进程读取不同的文件  如果读取数据的线程数比文件数还大，多个线程可能会读取到同一个文件中相近部分的数据，
导致过多的硬盘寻址，导致效率降低


'''
