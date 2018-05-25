import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split #用来区分训练集和测试集
from keras.models import Sequential  #导入keras
from keras.optimizers import Adam  
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

#每次运行代码，生成随机数一样
np.random.seed(0)


def load_data(args):
    """
    导入数据，并分成训练集和测试集
    """
    #利用panda 读取csv文件
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #三张照片路径作为输入样本，后续会随机选择一张作为输入数据
    X = data_df[['center', 'left', 'right']].values
    #对应方向盘转角为标签
    y = data_df['steering'].values

    #使用scikit learn包，划分数据集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    使用NVIDIA 推荐的模型，具体参数如下：
    
    Convolution Layer: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution Layer: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution Layer: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution Layer: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution Layer: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected Layer: neurons: 100, activation: ELU
    Fully connected Layer: neurons: 50, activation: ELU
    Fully connected Layer: neurons: 10, activation: ELU
    Fully connected Layer: neurons: 1 (output)

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))  #输入数据归一化，有助于收敛
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))  #激活函数采用ELU
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))  
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    模型训练
    """
    #每个epoch后保存模型
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #损失函数采用MSE loss，优化器采用ADAM,学习率为1.0e-4
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #由于电脑内存限制，采用batch_generator产生批训练数据，一共训练10个epoch，每个epoch有20000个样本
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)


def s2b(s):
    """
    将字符串类型转换成布尔类型
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
   载入数据，开始训练
    """
    parser = argparse.ArgumentParser(description='Autonomous Driving Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()


    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #载入data
    data = load_data(args)
    model = build_model(args)
    #开始训练并保存模型
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

