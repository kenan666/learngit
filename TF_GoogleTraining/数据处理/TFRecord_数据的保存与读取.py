import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#  将输入转化成TFRecord格式并保存
# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example

# 读取mnist训练数据。
mnist = input_data.read_data_sets("../../datasets/MNIST_data",dtype=tf.uint8, one_hot=True)
images = mnist.train.images

#  训练数据所对应的的  正确答案  ， 可以作为一个属性保存在TFRecord  中
labels = mnist.train.labels

#  训练数据的图像分辨率，这样可以作为Example  中的一个属性
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出包含训练数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output.tfrecords") as writer:
    for index in range(num_examples):
        example = _make_example(pixels, labels[index], images[index])
        writer.write(example.SerializeToString())
print("TFRecord训练文件已保存。")

# 读取mnist测试数据。
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(
            pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试文件已保存。")

#  读取TFRecord文件
# 读取文件。  创建一个  reader  来读取  TFRecord  中的文件数据
reader = tf.TFRecordReader()

#  创建一个队列来维护输入文件列表  tf.train.string_input_producer 函数
filename_queue = tf.train.string_input_producer(["output.tfrecords"])

#  从文件中  读取一个样例 ，   read_up_to 可以读取样例
_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })

#   tf.decode_raw  可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])