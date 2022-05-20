import os
from collections import OrderedDict as ODict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from albumentations import *
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import argparse
import logging
from typing import Dict, List, Any, Tuple


def load(data_dir: str = "/data") -> Tuple[x_train, y_train]:
    # locate, reshape and separate data into training and validation sets
    datadir = os.getcwd() + data_dir

    filenames = ["train.csv", "test.csv"]
    datadict = ODict()
    for files in filenames:
        try:
            with open(datadir + "/" + files, mode="r") as csvfile:
                datadict[files] = np.loadtxt(csvfile, delimiter=",", skiprows=1)
                csvfile.close()
            logging.info("file acquired: ./{}".format(files))
        except FileNotFoundError:
            logging.critical("file will be skipped ./{}".format(files))
    logging.info(datadict.keys(), filenames)

    trainmnist = datadict[filenames[0]]
    testmnist = datadict[filenames[-1]]

    train_labels = trainmnist[:, 0].reshape(-1)
    trainmnist = trainmnist[:, 1:].reshape(-1, 28, 28)
    testmnist = testmnist.reshape(-1, 28, 28)
    print(trainmnist.shape, train_labels.shape, testmnist.shape)

    x_train, x_test, y_train, y_test = train_test_split(trainmnist, train_labels,
                                                        test_size=0.2)
    return x_train, y_train


# Albumentations Set up
class AlbumData(Dataset):
    """
        This class receives both training sets and the set of transformations required and then
        turns them into a tensor
        x_data: x_train
        y_data: y_train
        transforms: None
    """

    # def __init__(self, x=x_train, y=y_train, transforms=None):
    def __init__(self, x, y, transforms=None):

        super().__init__()

        self.x = x
        self.y = y
        self.transform = transforms

        if self.y is None:
            self.len = len(self.x)
        else:
            try:
                assert len(self.x) == len(self.y)
                self.len = len(self.x)
            except AssertionError:
                print(f" the size of x ({len(self.x)} is different from y ({len(self.y)})")

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.y is not None:
            image, label = self.x[item], self.y[item]
            image = np.expand_dims(image, -1).astype(np.uint8)
            label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        else:
            image, label = self.x[item], None

        if self.transform is not None:
            aug_img = self.transform(image=image)
            image = aug_img["image"].astype(np.uint8).reshape(28, 28, 1)

        image = transforms.ToTensor()(image)

        if self.y is None:
            return image
        else:
            return image, label


# class AlbumDataPair(AlbumData):
#     def __getitem__(self, item):
#         image = super(AlbumDataPair, self).__getitem__()
#         label = ...

def main():
    load("/data")
    albumentations_transform = Compose([ShiftScaleRotate(shift_limit=0.11,
                                                         scale_limit=0.1,
                                                         rotate_limit=30,
                                                         interpolation=cv2.INTER_LANCZOS4,
                                                         border_mode=cv2.BORDER_CONSTANT,
                                                         p=0.75),
                                        OneOf([OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,
                                                                 p=1.0),
                                               GridDistortion(border_mode=cv2.BORDER_CONSTANT,
                                                              p=1.0)],
                                              p=0.75),
                                        Normalize(mean=[0.1307], std=[0.3081])])
    albumentations_valtransform = Compose([Normalize(mean=[0.1307], std=[0.3081])])
    load()
    test_set = AlbumData(x_train, y_train, transforms=albumentations_transform)
    fig, axes = plt.subplots(5, 5, figsize=(8, 8), sharex="all", sharey="all")
    _plots = None

    # test visualization
    testnum = np.random.randint(0, 28000)
    for axs in axes:
        for ax in axs:
            data = test_set.__getitem__(testnum)
            imgset = ax.imshow(data[0].reshape(28, 28), cmap='gray')
            imgset = ax.set_title(str(data[1].numpy()))
            imgset = ax.set_axis_off()

    plt.subplots_adjust(hspace=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
