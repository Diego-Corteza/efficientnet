import json
import logging
import albumentations
import torch
import numpy as np
from sklearn.datasets import fetch_openml
from typing import List, Dict, Tuple, Callable
from sklearn.model_selection import train_test_split
from data_logger import AlbumData

from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader


class Training:

    """
    The class 'train_model', which inherits from the class 'Model', is initialized with a dictionary that contains some parameters for the training:

    - lr:         Learning rate (default = 0.001).
    - epochs:     Number of epochs to train (default = 100).
    - MAX_EPOCHS: Max number of epochs allowed (default = 1000).

    And contains six methods:

    - train
    - classify
    - evaluate_performance
    - split_data
    
    """

    def __init__(self, parameters_file: str):

        super().__init__()

        training_parameters = json.load(open(parameters_file))

        self.__model = self.create_model(training_parameters["model"])
        print(f"model: {self.__model}")
        logging.info("creating data loader")
        self.__training_dataloader = self.create_train_dataloader(training_parameters["data_loader_def"])
        print("created data loader...")

        try:
            self.batch_size = training_parameters["training_params"]["batch_size"]
        except KeyError:
            self.batch_size = 16

        print("so far so good")
        # self.__preprocessor: [torch.nn.Module, None] = None
        # self.__device = torch.device("cuda") if torch.cuda.is_available() else
        self.__device = torch.device("cpu")

        try:
            self.lr = training_parameters["training_params"]['lr']
            self.epochs = training_parameters["training_params"]['epochs']
            self.MAX_EPOCHS = training_parameters["training_params"]['MAX_EPOCHS']
        except (KeyError, ValueError):
            self.lr = 0.001
            self.epochs = 10
            self.MAX_EPOCHS = 100

        print("doneee")

    def create_model(self, model_params) -> torch.nn.Module:
        model_type = model_params["type"]
        if model_type == "efficientnet-b0":
            # import maravillas model
            pass
        model: torch.nn.Module = EfficientNet.from_pretrained('efficientnet-b0')
        return model

    def process_yaml(self):
        # get optimzier and its parameters
        # epochs, lr
        # criterion to use
        'Nada por aqui'

    def __call__(self, *args, **kwargs):
        self.train()
        return True

    def finalize(self):
        """
        closes all loggers
        makes sure that the final model is saved

        :return:
        """
        self.t_board_logger.close()

    def __update_preprocessor(self, preprocessor: torch.nn.Module) -> None:
        self.__preprocessor = preprocessor

    def create_train_dataloader(self, dataloader_def):
        dataset = AlbumData()
        dataloader = DataLoader(dataset, batch_size=1) # self.batch_size, shuffle=True)
        return  dataloader

    @staticmethod
    def create_train_dataloader_v2():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, Y = mnist["data"], mnist["target"]

        X_t = torch.from_numpy(X).float().cuda()
        Y_t = torch.from_numpy(Y.astype(int)).long().cuda()

        X_train, X_test, y_train, y_test = split_data(X_t, Y_t)
        return ((X_train, y_train))

    @staticmethod
    def create_test_dataloader( yaml_data_def: dict):
        augmentation_list = albumentations.load("settings.yaml", data_format="yaml")
        return DataLoader(AlbumData(transforms=augmentation_list), batch_size=32)

    def _preprocess(self, x: torch.TensorType) -> None:
        if self.__preprocessor:
            x = self.__preprocessor(x)
        return x


    def get_data(self):
        pass


    @staticmethod
    def cross_entropy(output, target):
        logits = output[torch.arange(len(output)), target]
        loss = - logits + torch.log(torch.sum(torch.exp(output), axis=-1))
        loss = loss.mean()
        return loss

    def train(self, *args, **kwargs):
        logging.info("entering training")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            self.__model.parameters(), self.lr, momentum=0.5, nesterov=True
        )

        epoch_loss = []
        epoch_batch = 10

        # images, gts = self.__training_dataloader
        # images.to(self.__device)
        # gts.to(self.__device)

        logging.info("about to start training")
        for epoch in range(self.epochs):

            loss = self.train_single_epoch(optimizer, criterion)
            epoch_loss.append(loss)

            # self.validate(images, gts)
            # self.test(images, gts)
            if epoch >= self.MAX_EPOCHS:
                break
        # self.finalize()
        logging.info("done training")
        return True


    def train_single_epoch(self, optimzier, criterion):
        """
        Method to train a model given a dataset.
        """

        self.__model.train()
        self.__model.to(self.__device)

        running_loss = 0.0
        num_batches = 1
        for data in self.__training_dataloader:

            # Gradiends are turned to zero in order to avoid acumulation
            self.__model.zero_grad()

            images, gts = data
            images.to(self.__device)
            gts.to(self.__device)

            # forward pass
            y_pred = self.__model(images)

            # loss
            train_loss = criterion(y_pred, gts)
            train_loss.backward()

            optimzier.step()

            running_loss += train_loss.item()
            num_batches += 1

            # tensorboard logging
            # mlflow logging
            print(train_loss)

        # Backprop
        # train_loss.backward()

        # # # updates
        # with torch.no_grad():
        #     for param in self.__model.parameters():
        #         param -= self.lr * param.grad

        # running_loss += train_loss.item()

        # # loss logger
        # self.log_metric("train_loss", train_loss)

        # if not epoch % epoch_batch:
        #     print(f"Epoch {epoch}/{self.epochs} Loss {np.mean(epoch_loss):.5f}")
        #
        # if epoch >= self.MAX_EPOCHS:
        #     break
        avg_loss = running_loss / num_batches
        return avg_loss


    def log_metric(self, metric_key: str, metric_value: float) -> None:
        #self.t_board_logger ...
        #self.mlflow_logger ...
        pass


    def validate_single_epoch(self):
        self.__model.val()
        with torch.no_grad():
            for data in self.__validation_dataloader:

                # get data
                images, gts = data
                preds = self.__model(images)

                # call metric logger


        # calculate/call accuracy metrics and logger


    def test(self):
        self.__model.val()
        # get data
        # call metric logger

        # calculate/call accuracy metrics and logger
            # confusion matrix
            # reconstruction images if auto encodeer

    def save_model(self):

        # torch.save(..)
        pass


    def classify(self):
        """
        Method to classify samples using the existing model.
        """


    def get_accuracy(self, x_data, y_data):
        """
        This method receives a set of samples (x_data), with their respective targets (y_data), and gets the accuracy of the model.
        """
        


    def split_data(self, test_percent=0.20, seed=42):
        """
        The method split_data(), which belongs to the class Model, receives four parameters:

        - x_data: Samples (torch.Tensor).
        - y_data: Targets of the samples (torch.Tensor).
        - test_percent: Percentage of total data employed for testing (default = 0.20).
        - seed:         Used to replicate the resulting training and evaluation sets (default = None).

        And it is going to return two sets one for training and one for testing (each of which contains is divided into samples and targets).
        """

        x_numpy = self.x_data.cpu().numpy()
        y_numpy = self.y_data.cpu().numpy()

        X_train, X_test, y_train, y_test = train_test_split(x_numpy, y_numpy, test_size=test_percent, random_state=seed)
        
        return torch.from_numpy(X_train).float().cuda(), torch.from_numpy(X_test).float().cuda(), torch.from_numpy(y_train).long().cuda(), torch.from_numpy(y_test).long().cuda()


class AutoEncoder(Training):
    pass
