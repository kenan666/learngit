import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#  原始数据的目录，这个目录下有  5  个子目录每个子目录下保存属于该类别的所有图片
INPUT_DATA = 'C:/Users/KeNan/Desktop/TF_Google/CH6_图像识别与卷积神经网络/flower_photos'

#  输出文件地址
OUTPUT_DATA = 'C:/Users/KeNan/Desktop/TF_Google/CH6_图像识别与卷积神经网络'

#  测试数据 和验证数据  比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

#  读取数据 并将 数据分割成  训练数据 ，验证数据  和  测试数据
def create_image_lists(sess,testing_percentage,validation_percentage):
    sub_dirs = [x[0] for i in os.work(INPUT_DATA)]
    is_root_dir = True
    
    #  初始化  各个  数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0
    
    #  读取所有子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取子目录中所有的图片文件
        extensions = ['jpg','jepg','JPG','JEPG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        
        for extensions in extensions:
            file_glob = os.path.join (INPUT_DATA,dir_name,'*.', + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        
        #  处理图像数据
        for file_name in file_list:
            #  读取并解析 图片  ，将图片转化为  299*299  ，方便使用inception-v3  模型来处理
            image_raw_data = gfile.FastGFile(file_name,'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image,dtype = tf.float32)
            image = tf.image.resize_images(images,[299,299])
            image_value = sess.run(image)
            
            #  随机划分数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append (image_value)
                testing_labels.append(current_label)
            else:
                training_images.append (image_value)
                training_labels.append(current_label)
        current_label += 1
        
    #  将训练数据  随机打乱  以获得更好的训练结果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    
    return np.asanyarray([training_images,training_labels,validation_images,validation_labels,
                         testing_images,testing_labels])

#  主函数
def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess,TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
        
        #  通过  numpy  格式保存处理后的数据
        np.save(OUTPUT_DATA,processed_data)
        
if __name__ == '__main__':
    main()