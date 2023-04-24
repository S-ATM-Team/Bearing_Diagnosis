import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from PIL import Image
from mymodels import *
from se_resnet import *


BATCH_SIZE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_directory(directory_name, height, width, normal):
    file_list = os.listdir(directory_name)
    file_list.sort(key=lambda x: int(x.split('-')[0]))
    img = []
    label0 = []

    for each_file in file_list:
        img0 = Image.open(directory_name + '/' + each_file)
        img0 = img0.convert('L')
        gray = img0.resize((height, width))
        img.append(np.array(gray).astype(np.float))
        label0.append(float(each_file.split('.')[0][-1]))
    if normal:
        data = np.array(img) / 255.0  # 归一化
    else:
        data = np.array(img)
    # data = data.reshape(-1, 1, height, width)
    label = np.array(label0)
    return data, label


class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

    def __getitem__(self, index):
        assert index < len(self.pics)
        return torch.Tensor([self.pics[index]]), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor([self.pics]), torch.Tensor(self.labels)


def read_data():
    read_data_start = time.time()
    height = 256
    width = 256
    # 小波时频图---2D-CNN输入
    x_train, y_train = read_directory(r'cwt_picture1\train', height, width, normal=1)
    x_valid, y_valid = read_directory(r'cwt_picture1\valid', height, width, normal=1)
    x_test, y_test = read_directory(r'cwt_picture1\test', height, width, normal=1)
    # x_train, y_train = read_directory(r'noise\cwt_0HPpicture\train', height, width, normal=1)
    # x_valid, y_valid = read_directory(r'noise\cwt_0HPpicture\valid', height, width, normal=1)
    # x_test, y_test = read_directory(r'noise\cwt_0HPpicture\test', height, width, normal=1)

    trainset = MyData(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    validset = MyData(x_valid, y_valid)
    valid_iter = torch.utils.data.DataLoader(
        validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = MyData(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    read_data_end = time.time()
    print("===============================> Time for reading data: {:.4f}s".format(read_data_end - read_data_start))

    return train_iter, valid_iter, test_iter


def train_and_valid(model, loss_function, optimizer, epochs):
    # device = torch.device("cpu")
    model.to(device)
    record = []
    best_acc = 0.0
    best_epoch = 0
    sum_time=0

    # train_data, valid_data = read_data()

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        epoch_time=0
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            # inputs=inputs.float().expand(-1,3,256,256)

            # layer = CBAMLayer(1)
            # inputs = layer.forward(inputs)

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.float32)


            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels.long())

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                # inputs = inputs.float().expand(-1,3,256,256)

                # layer = CBAMLayer(1)
                # inputs = layer.forward(inputs)

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels.long())

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        train_data_size = len(train_data) * BATCH_SIZE
        valid_data_size = len(valid_data) * BATCH_SIZE

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        epoch_time = epoch_end-epoch_start
        sum_time+=epoch_time

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, "
              "Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
            epoch_end - epoch_start))
        print("* Best Accuracy for validation : {:.4f}% at epoch {:03d}".format(best_acc * 100, best_epoch))
    print('Average Time: {:.4f}s'.format(sum_time/num_epochs))


    return model, record


if __name__ == '__main__':
    num_epochs = 50
    model =cbam2_ResNetmodel()
    # loss_func = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.fc.parameters(),lr=0.01,weight_decay=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_data, valid_data, test_data = read_data()
    trained_model, record = train_and_valid(model, loss_func, optimizer, num_epochs)
    torch.save(trained_model.state_dict(), r'output\model\ResNet\cbam2_ResNetmodel.pth')

    record = np.array(record)
    plt.plot(record[:, 0:2])
    # plt.legend(['Train Loss', 'Valid Loss'])
    plt.legend(['训练损失', '验证损失'], prop={'family': 'SimHei', 'size': 10})
    plt.xlabel('迭代次数', fontproperties={'family': 'SimHei', 'size': 10})
    plt.ylabel('损失', fontproperties={'family': 'SimHei', 'size': 10})
    plt.show()

    plt.figure(figsize=(5, 5), dpi=240)
    plt.plot(record[:, 2:4])
    plt.legend(['训练精度', '验证精度'],prop={'family':'SimHei','size':10})
    plt.xlabel('迭代次数',fontproperties={'family':'SimHei','size':10})
    plt.ylabel('准确率',fontproperties={'family':'SimHei','size':10})
    # plt.ylim(0, 1)
    # plt.savefig('accuracy.png')
    plt.show()