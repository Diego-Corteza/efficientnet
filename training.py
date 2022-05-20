import json
import torch
import numpy as np
from sklearn.datasets import fetch_openml
from typing import List, Dict, Tuple, Callable
from sklearn.model_selection import train_test_split

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

    def __init__(self, model, parameters_file: str):

        super().__init__()

        training_parameters = json.load(open(parameters_file))

        self.__model = model
        self.__training_dataloader = self.create_train_dataloader()
        # self.__preprocessor: [torch.nn.Module, None] = None
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.lr = training_parameters['lr']
        self.epochs = training_parameters['epochs']
        self.MAX_EPOCHS = training_parameters['MAX_EPOCHS']


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

    @staticmethod
    def create_train_dataloader():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, Y = mnist["data"], mnist["target"]

        X_t = torch.from_numpy(X).float().cuda()
        Y_t = torch.from_numpy(Y.astype(int)).long().cuda()

        X_train, X_test, y_train, y_test = split_data(X_t, Y_t)
        return ((X_train, y_train))

    @staticmethod
    def create_test_dataloader( yaml_data_def: dict):
        augmentation_list: List = yaml_data_def["test_augmentations"]
        return DiegosDataloader(augmentation_list)

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

    def train(self):
        """
        Method to train a model given a dataset.
        """

        self.__model.train()
        images, gts = self.__training_dataloader
        images.to(self.__device)
        gts.to(self.__device)

        # x = self._preprocess(x)

        epoch_loss = []
        epoch_batch = 10

        for epoch in range(1, self.epochs+1):  

            y_pred = self.__model(images)

            # Gradiends are turned to zero in order to avoid acumulation
            self.__model.zero_grad()

            # loss
            train_loss = self.cross_entropy(y_pred, gts)
            epoch_loss.append(train_loss.item())

            # Backprop
            train_loss.backward()

            # # updates
            with torch.no_grad():
                for param in self.__model.parameters():
                    param -= self.lr * param.grad

            # running_loss += train_loss.item()

            # # loss logger
            # self.log_metric("train_loss", train_loss)

            if not epoch % epoch_batch:
                print(f"Epoch {epoch}/{self.epochs} Loss {np.mean(epoch_loss):.5f}") 

            if epoch >= self.MAX_EPOCHS:
                break

        return True


    def log_metric(self, metric_key: str, metric_value: float) -> None:
        #self.t_board_logger ...
        #self.mlflow_logger ...
        pass


    def validate(self):
        self.__model.val()
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
