import tensorflow as tf

#  创建文件列表，通过文件列表创建输入文件队列，读取文件
files = tf.train.match_filenames_once("output.tfrecords")
filename_queue = tf.train.string_input_producer(files, shuffle=False) 

#  解析TFRecord文件里的数据
# 读取文件。

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })

#  从原始图像数据解析出 像素矩阵 ，并根据尺寸  还原图像
decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'],tf.int32)

#pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])

# 将文件以100个为一组打包
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([images, labels], 
                                                    batch_size=batch_size, 
                                                    capacity=capacity, 
                                                    min_after_dequeue=min_after_dequeue)

# 训练模型
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2

# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)
    
# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
                  
        sess.run(train_step) 
    coord.request_stop()
    coord.join(threads)       


## 第二种写法

#  创建文件列表，并通过文件列表创建输入文件队列
files = tf.train.match_filename_once("C:/Users/KeNan/Desktop/TF_Google/CH7_图像数据处理/file_pattern-*")
filename_queue = tf.train.string_input_producer(files ,shuffle = False)

#  解析 TFRecord中的数据
#  假设image中存储的是图像的原始数据，label 为该样例对应的标签，height，width channel  给出的是图像的维度
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                  features = {
                                      'image':tf.FixedLenFeature([],tf.string),
                                      'label':tf.FixedLenFeature([],tf.int64),
                                      'height':tf.FixedLenFeature([],tf.int64),
                                      'width':tf.FixedLenFeature([],tf.int64),
                                      'channels':tf.FixedLenFeature([],tf.int64)
                                  })
image,label = features['image'],features['label']
height,width = features['height'],features['width']
channels = features['channels']

#  从原始图像数据 解析出像素矩阵  ，并根据像素尺寸还原图像
decoded_image = tf.decode_raw(image,tf.uint8)
decoded_image.set_shape = ([height,width,channels])

#  定义神经网络输入层 图片的大小
image_size = 299

#  preprocess_for_train  预处理
distorted_image = preprocess_for_train(decoded_image,image_size,image_size,None)

#  将处理后的图像和标签数据  通过  tf.train.shuffle_batch  整理成神经网络训练时需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch,label_batch = tf.train.shuffle.batch([distorted_image,label],batch_size = batch_size,capacity = capacity,
                                                 min_after_dequeue = min_after_dequeue)

#  定义神经网络的结构以及优化过程
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#  声明会话以及运行神经网络的优化过程
with tf.Session() as sess:
    # 初始化变量，启动线程
    sess.run(tf.global_variables_initializer(),tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    # 训练
    TRAINING_ROUND = 5000
    for i in range(TRAINING_ROUND):
        sess.run(train_step)
    
    #  停止线程
    coord.request_stop()
    coord.close(threads)