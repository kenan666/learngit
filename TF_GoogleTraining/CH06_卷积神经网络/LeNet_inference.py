import tensorflow as tf

#  配置参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#  第二层 卷积层  尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#  全连接层的节点数
FC_SIZE = 512

#  定义卷积神经网络的 前向传播过程，添加一个新的参数train，用于区分训练过程和测试过程和测试过程。
#  dropout 方法  ---》dropout可以进一步提升模型可靠性防止过拟合，且只在训练时使用。
def inference(input_tensor, train, regularizer):
    #  声明第一层卷积神经网络的变量和实现前向传播过程
    #  此处定义的 卷积层 为  28*28*1的原始 MNIST 图片像素 --》输出为  28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        
        #  使用 边长  为 5 ，深度为32 的过滤器 ，过滤器的步长为 1 ，
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    
    # 实现第二层池化层  前向传播过程  --》此处选用的是最大池化层，边长为 2 
    #  这一层的输入是上一层的 输出，--》28*28*32  -》此处输出为 14*14*32
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
    
    #  声明第三层卷积神经网络的变量  并实现前向传播 过程  ，这一层输入  为  14*14*32  输出为--》14*14*64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        
        #  使用边长为 5 深度为 64 的 过滤器，过滤器移动步长为  1 
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    #  实现第四层  池化层的前向传播过程。  这一层与第二层结构是一样的，输入为--》14*14*64，输出为--》7*7*64
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #  将第四层池化层的输出转化为  第五层全连接层的  输入格式。第四层中输出为  7*7*64  ，        
        pool_shape = pool2.get_shape().as_list()
        
        # 计算 将矩阵  拉直 成为向量之后的长度，  长度 =  长*宽*深度   
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        
        #  通过tf.reshape() 函数将第四层的输出变成一个batch 的向量
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    
    #  声明第五层全连接层的变量并实现  前向传播过程
    #  输入为拉直之后的一组向量  输入 3136  --》  输出  512
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        #  加入  正则化
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
    
    #  声明第六层全连接层的变量并实现前向传播过程
    #  输入为 第五层输出  512  ---》  输出 为   10    --》  通过  softmax  得到最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit