'''
文件的训练量过大时，可以利用多个 TFRecord 来进行管理
tf  中还提供了  tf.train.match_filenames_once  函数来获取符合一个正则表达式的所有文件，得到的文件列表可以通过  
tf.train.string_input_producer 函数来进行管理

tf.train.string_input_producer  函数会使用初始化时提供的文件列表创建一个输入队列，输入队列中原始的元素问文件列表中的所有文件
tf.train.string_input_producer  函数支持随机打乱文件列表中文件出队顺序  通过设置shuffle参数  
    设置为 true 时  --》文件加入队列之前会被打乱顺序，出队也是随机的

tf.train.string_input_producer 函数可以设置 num_epochs 参数来限制加载出示文件列表的最大轮数 
--测试神经网络的时候，所有测试数据需要使用一次，可以将  num_epochs 设置为 1  ，计算 一轮之后，自动停止
'''

#  tf.train.match_filename_once  函数使用
import tensorflow as tf

#  创建  TFRecord  文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

# 模拟海量数据情况下将数据写入不同文件   利用 num_shards  定义写入的文件数  instances_per_shard  定义每个文件中多少数据
num_shards = 2
instances_per_shard = 2
for i in range(num_shards):
    #  将数据分为多个文件时，可以将不同的文件以类似 0000n - of - 0000m  的后缀区分
    filename = ('C:/Users/KeNan/Desktop/TF_Google/CH7_图像数据处理/data.tfrecords-%.5d-of-%.5d'% (i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    
    #将数据封装成 Example 结构，并写入TFRecord文件
    for j in range(instances_per_shard):
        #  example 结构仅包含当前样例属于第几个文件以及当前文件的第几个样本
        example = tf.train.Example(features = tf.train.Features(feature = {
            'i':_int64_feature(i),
            'j':_int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()

#  tf.train.string_input_producer  使用方法

#  使用tf.train.match_filenames_once  函数  获取文件列表
files = tf.train.match_filenames_once("C:/Users/KeNan/Desktop/TF_Google/CH7_图像数据处理/data.tfrecords-*")

#  通过tf.train.string_input_producer 函数创建输入队列，文件列表为  tf.train.match_filenames_once  函数获取的文件列表
#  shuffle  此处设置为 false  实际情况下  设置为 true
filename_queue = tf.train.string_input_producer(files, shuffle=False) 
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      })
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print (sess.run(files))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        print (sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)