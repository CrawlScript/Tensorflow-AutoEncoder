from mpl_toolkits.mplot3d import Axes3D
from autoencoder import AutoEncoder, DataIterator
import codecs
from random import shuffle
from matplotlib import pyplot as plt
import numpy as np

class IrisDataSet(object):

    def get_label_id(self, label):
        if label in self.label_id_dict:
            return self.label_id_dict[label]
        self.label_id_dict[label] = self.next_label_id
        self.next_label_id += 1
        return self.next_label_id - 1

    def __init__(self):
        self.next_label_id = 0
        self.label_id_dict = {}
        with codecs.open("tutorial_datasets/iris/iris.data", "r", "utf-8") as f:
            str_datas = [line.strip() for line in f]
        str_datas = [line.split(",") for line in str_datas if len(line) > 0]
        shuffle(str_datas)
        self.datas = [[float(d) for d in row_data[0:-1]] for row_data in str_datas]
        # normalize datas
        self.datas = np.array(self.datas, dtype = np.float64)
        self.datas = self.datas/self.datas.max(0)

        self.labels = [self.get_label_id(row_data[-1]) for row_data in str_datas]

iris_dataset = IrisDataSet()
# train data
datas = iris_dataset.datas
labels = iris_dataset.labels

# data wrapper
iterator = DataIterator(datas)

# train autoencoder
# assume the input dimension is input_d
# the network is like input_d -> 4 -> 2 -> 4 -> input_d
autoencoder = AutoEncoder()
autoencoder.fit([4, 2], iterator, learning_rate = 0.02, max_epoch = 10000)

# after training

# encode data
encoded_datas = autoencoder.encode(datas)
print "encoder ================"
print encoded_datas 

#visualize encoded datas
colors = ["red", "green", "blue"]
label_colors = [colors[label_id] for label_id in labels]

fig_3d =plt.figure("origin iris data")
plot_3d = fig_3d.add_subplot(111, projection='3d')
plot_3d.scatter(datas[:,0], datas[:,1], datas[:, 2], color = label_colors)

fig_2d = plt.figure("encoded iris data")
plot_2d = fig_2d.add_subplot(111)
plot_2d.scatter(encoded_datas[:,0], encoded_datas[:,1], color = label_colors)

plt.show()



