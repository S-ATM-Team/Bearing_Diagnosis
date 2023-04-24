import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from mymodels import *
from se_resnet import *



BATCH_SIZE = 7
SCALE = 448

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
    num_classes = 10
    height = 256
    width = 256

    # x_test, y_test = read_directory(r'cwt_3HPpicture\all', height, width, normal=1)
    x_test, y_test = read_directory(r'cwt_picture1\test', height, width, normal=1)
    # x_test, y_test = read_directory(r'noise\cwt_0HPpicture\0dBtest', height, width, normal=1)


    testset = MyData(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    read_data_end = time.time()
    print("===============================> Time for reading data: {:.4f}s".format(read_data_end - read_data_start))
    return test_iter


def test():
    print("loading model ...")
    load_model_start = time.time()
    model.load_state_dict(torch.load(r'output\model\ResNet\cbam_ResNetmodel.pth'))
    model.eval()

    load_model_end = time.time()
    print("time for loading model: {:.4f}s \n".format(load_model_end - load_model_start))

    model.to(device)

    epoch_start = time.time()

    correct = 0
    total = 0
    y_test = []
    y_pre = []
    y_out = []
    with torch.no_grad():
        model.eval()

        for k, (inputs, labels) in enumerate(test_data):
            # inputs = inputs.float().expand(-1, 3, 256, 256)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            ret, predictions = torch.max(outputs.data, 1)
            total += BATCH_SIZE
            correct += (predictions == labels).sum().item()

            outputs = outputs.cpu()
            labels = labels.cpu()
            predictions = predictions.cpu()

            y_out.append(outputs)
            y_test = np.append(y_test, labels)
            y_pre = np.append(y_pre, predictions)
        # 打印准确率
        print(correct / total * 100)

    epoch_end = time.time()
    print(" Time: {:.4f}s".format(epoch_end - epoch_start))
    # 绘制混淆矩阵
    con_mat = confusion_matrix(y_test.astype(str), y_pre.astype(str))
    print(con_mat)
    classes = list(set(y_test))
    classes.sort()

    plt.figure(dpi=240)
    plt.imshow(con_mat, cmap=plt.cm.cool)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('预测标签',fontproperties={'family':'SimHei'})
    plt.ylabel('真实标签',fontproperties={'family':'SimHei'})
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.show()

    # 分类可视化
    y_out = np.vstack(y_out)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(y_out[:])
    # -------------------------------可视化--------------------------------
    # y_test_cat = np_utils.to_categorical(y_test[:2400], num_classes=10)# 总的类别
    plt.figure(figsize=(10, 10))
    color_map = y_test[:]
    for cl in range(10):  # 总的类别
        indices = np.where(color_map == cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=cl)
        # plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=cl)
    plt.tick_params(labelsize=18)
    plt.legend(prop= {'size': 13},loc='upper left')
    plt.savefig(r't_sne_end', dpi=600)
    plt.show()


if __name__ == '__main__':
    model = cbam_ResNetmodel()
    test_data = read_data()
    test()