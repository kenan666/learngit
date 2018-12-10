import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

#  加载inception-v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

#  处理好之后的数据文件
INPUT_DATA = 'C:/Users/KeNan/Desktop/TF_Google/CH6_图像识别与卷积神经网络'

#  保存训练好的模型的路径
TRAIN_FILE = 'C:/Users/KeNan/Desktop/TF_Google/CH6_图像识别与卷积神经网络'

#  训练好的模型的地址
CKPT_FILE = 'C:/Users/KeNan/Desktop/TF_Google/CH6_图像识别与卷积神经网络'

#  定义训练中使用的参数
LEARNING_RATE = 0.0001
STEP = 300
BATCH = 32
N_CLASSES = 5

#不需要从训练好的模型中加载的参数   此处为最后的全连接层，在新的一层中需要重新训练这一层中的参数。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/Auxlogits'

#  需要训练的网络参数的名称  ，在fine-tuning  的过程中就是最后的全连接层
TRAINING_SCOPES = 'InceptionV3/Logits,InceptionV3/Auxlogits'

#  获取所有的训练好的模型中的参数
def get_tuned_avriables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variable_to_restore = []
    
    #  枚举inception-v3  模型中所有的参数，然后判断  是否需要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    return variable_to_restore

#  获取所需要训练的变量列表
def get_trainable_variable():
    scopes = [scope.strip() for scope in TRAINING_SCOPES.split(',')]
    variables_to_train = []
    
    #  枚举所需要训练的参数前缀，并通过这些前缀找到所有参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main():
    #  加载处理好的数据
    processed_data = np.load(INPUT_DATA)

    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]

    validation_images = processed_data[2]
    validation_labels = processed_data[3]

    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    print ("%d training examples, %d validation examples and %d testing examples." % (n_training_example,len(validation_labels),len(testing_labels)))

    #  定义 inception-v3  的输入，images为输入图片，labels为每一张图片的标签
    images = tf.placeholder(tf.float32,[None,299,299,3],name = 'input_images')
    labels = tf.placeholder(tf.int64,[None],name = 'labels')

    #  定义inception-v3 模型  
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits , _ = inception_v3.inception_v3(images,num_classes=N_CLASSES)
    
    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()

    #  定义交叉熵损失，---》模型在定义的时候已经将正则损失加入损失集合了
    tf.losses.softmax_cross_entropy(tf.one_hot(labels,N_CLASSES),logits,weights = 1.0)
    
    #  定义训练过程
    train_step = tf.train.RMpropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    #  计算正确率
     
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #  定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE,get_tuned_avriables(),ignore_missing_vars=True)

    # 定义保存新的训练好的模型的函数
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #  初始化没有加载进来的变量  ，---》此过程在模型加载之前，否则初始化过程会将加载好的变量重新赋值
        init = tf.global_variables_initializer()
        sess.run(init)

        #  加载训练好的模型
        print ('Loading tuned variables from %s ' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            # 运行训练过程，--》此时不会更新全部的 参数，，只更新指定的参数
            sess.run(train_step,feed_dict = {
                images:training_images[start:end],
                labels:training_labels[start:end]})

            # 输出日志
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess,TRAIN_FILE,global_step = i)
                validation_accuracy = sess.run(evaluation_step,feed_dict = {images:validation_images,labels:validation_labels})

                print ('Step %d :Validation accuracy = %.lf%%' %(i,validation_accuracy * 100.0))

            #  使用训练数据
            start = end 
            if start == n_training_example:
                start = 0

            end = start + BATCH
            if end > n_training_example:
                end = n_training_example

            # 在测试数据上  测试正确率
            test_accuracy = sess.run(evaluation_step,feed_dict = {images:testing_images,labels:testing_labels})
            print ('Final test accuracy = %.lf%%' % (test_accuracy * 100))

if __name__ == '__main__':
    tf.app.run()    