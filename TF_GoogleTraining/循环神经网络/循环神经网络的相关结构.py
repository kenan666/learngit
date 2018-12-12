import tensorflow as tf

#  定义一个 LSTM 结构 LSTM  中使用的变量也会在函数中自动被声明

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#  将LSTM中的状态初始化为全 0 数组  
state = lstm.zero_state(batch_size,tf.float32)

#  定义损失函数
loss = 0.0

#  测试时，循环神经网络可以处理任意长度的序列，训练时，为了将循环神经网络展开成前馈神经网络，
# 需要知道训练数据的序列长度
for i in range (num_step):
    #  在第一个时刻声明LSTM 结构中使用的变量，  之后需要复用之前定义好的变量
    if i > 0 :
        tf.get_variable_scope().reuse_variable()
        
        #  处理  时间序列中的一个时刻
        lstm_output,state = lstm(current_input,state)
        
        #  将当前时刻LSTM结构的输出传入  一个全连接层得到最后的输出
        final_output = fully_connected(lstm_output)
        
        #计算当前时刻输出的损失
        loss += calc_loss (final_output,excepted_output)

#  训练模型与之前类似

#******************************************************************************************************
#  深层循环神经网络  Deep RNN

#  定义一个基本的  LSTM  结构作为循环体的基础结构
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell()

#  通过MultiRNNCell 类实现 深层循环神经网络  中每一个时刻的前向传播过程
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size) for _ in range(number_of_layers)])

#  与经典的循环神经网络一样， 通过zero_state 函数来获取初始状态
state = stacked_lstm.zero_state(batch_size,tf.float32)

#  计算每一时刻的前向传播结果
for i in range(num_steps):
    if i > 0 :
        tf.get_variable_scope().reuse_variable()
        stacked_lstm_output,state = stacked_lstm(current_input,state)
        final_output = fully_connected(stacked_lstm_output)
        loss += calc_loss(final_output,excepted_output)


# ********************************************************************************************************
#  循环神经网络中  dropout  的方法的实现
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell

#  使用  DropoutWrapper类实现dropout功能
'''
dropout  中有两个参数 
1-- input_keep_prob  控制输入的dropout概率
2-- output_keep_prob  控制输出dropout概率
'''
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size)) for _ in range(number_of_layers)])