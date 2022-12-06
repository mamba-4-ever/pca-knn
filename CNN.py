import tensorflow as tf
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

train_path = r'./Image Database/Image Database/trainset/'
x_train_savepath = './Image Database/Image_x_train.npy'
y_train_savepath = './/Image Database/Image_y_train.npy'

test_path = r'./Image Database/Image Database/testset/'
x_test_savepath = './Image Database/Image_x_test.npy'
y_test_savepath = './Image Database/Image_y_test.npy'

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


def generateds(path):
    x, y_ = [], []  # 建立空列表
    #读取文件夹
    for subdirname in os.listdir(path):
        subjectpath = os.path.join(path, subdirname)
        if os.path.isdir(subjectpath):
            # 一个文件夹一个人照片

            for filename in os.listdir(subjectpath):
                img_path = os.path.join(subjectpath, filename)#读取图片
                img = Image.open(img_path)  # 读入图片
                img = img.resize((28,28), Image.ANTIALIAS)
                img = np.array(img.convert('L'))  # 图片变为的np.array格式
                img = img / 255.  # 数据归一化 （实现预处理）
                x.append(img)  # 归一化后的数据，贴到列表x
                y_.append(subdirname)  # 标签付给y_

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_
#查看参数是否计算
if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath) #读取已有参数
    y_train = np.load(y_train_savepath) #读取已有参数
    x_test_save = np.load(x_test_savepath) #读取测试参数
    y_test = np.load(y_test_savepath) #读取测试参数
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28)) #参数矩阵化
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28)) #参数矩阵化
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path) # 建立训练集
    x_test, y_test = generateds(test_path) #建立测试集
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，使数据和网络结构匹配
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

model = Baseline()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
