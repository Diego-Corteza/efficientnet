import os
from collections import OrderedDict as ODict
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.utils.data.dataset as dataset
from data_logger import AlbumData

# locate, reshape and separate data into training and validation sets
datadir = os.getcwd() + "/data"
filenames = ["train.csv", "test.csv"]
datadict = ODict()
for files in filenames:
    try:
        with open(datadir + "/" + files, mode="r") as csvfile:
            datadict[files] = np.loadtxt(csvfile, delimiter=",", skiprows=1)
            csvfile.close()
        print("file acquired: ./{}".format(files))
    except FileNotFoundError:
        print("file will be skipped ./{}".format(files))
print(datadict.keys(), filenames)

trainmnist = datadict[filenames[0]]
testmnist = datadict[filenames[-1]]

train_labels = trainmnist[:, 0].reshape(-1)
trainmnist = trainmnist[:, 1:].reshape(-1, 28, 28)
testmnist = testmnist.reshape(-1, 28, 28)
print(trainmnist.shape, train_labels.shape, testmnist.shape)

x_train, x_test, y_train, y_test = train_test_split(trainmnist, train_labels,
                                                    test_size=0.2)
x_mnist = testmnist

for idx in [x_train, x_test, y_train, y_test, x_mnist]:
    print(f'Shape: {idx.shape}')

fig, ax = plt.subplots(1, 3, sharex="all", squeeze=True)
for img, x in zip(ax, [x_train, x_test, x_mnist]):
    img.set_axis_off()
    mnist_set = img.imshow(x[-1], cmap="gray")
plt.show()

test_set = AlbumData()
img, labl = test_set.__getitem__(0)
print(f"shape: {img.shape}, target: {labl}")
imgset1 = plt.imshow(img.reshape(28, 28), cmap="gray")
imgset1 = plt.axis