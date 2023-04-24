import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.io import loadmat
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

def prepro(d_path, length, number, normal, rate):

    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    def capture(original_path):
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]

            samp_train = int(number * (1 - slice_rate))  # 1000(1-0.3)
            Train_sample = []
            Test_Sample = []

            for j in range(samp_train):
                sample = slice_data[j*1500: j*1500 + length]
                Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                sample = slice_data[samp_train*1500 + length + h*1500: samp_train*1500 + length + h*1500 + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        data_all = np.vstack((Train_X, Test_X))
        scalar = preprocessing.StandardScaler().fit(data_all)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):

        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)

        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]

            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)

    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
path = r'data/0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = prepro(
    d_path=path,
    # length=784,
    length=2048,
    number=70,
    normal=True,
    rate=[0.6, 0.2, 0.2])

def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / 2048
    npower = xpower / snr
    random.seed(1)
    noise1 = np.random.randn(2048) * np.sqrt(npower)
    return x + noise1


plt.figure()
t = np.arange(0, 2048, 1)

n = wgn(x_train[150], 3)

plt.figure(figsize=(20, 20),dpi=80)
plt.subplot(511)
plt.plot(t, list(x_train[150]),label='原始信号')
plt.legend(prop={'family':'SimHei','size':20},loc='upper left')
# plt.title('The original signal-x')

plt.subplot(512)
plt.plot(t, wgn(x_train[150], -2),label='-2dB')
plt.legend(prop={'family':'SimHei','size':20},loc='upper left')
# plt.title('The original sinal with Gauss White Noise')

plt.subplot(513)
plt.plot(t, wgn(x_train[150], 0),label='0dB')
plt.legend(prop={'family':'SimHei','size':20},loc='upper left')

plt.subplot(514)
plt.plot(t, wgn(x_train[150], 2),label='2dB')
plt.legend(prop={'family':'SimHei','size':20},loc='upper left')

plt.subplot(515)
plt.plot(t, wgn(x_train[150], 10),label='10dB')
plt.legend(prop={'family':'SimHei','size':20},loc='upper left')

plt.tight_layout()
print(y_train[150])
plt.show()



