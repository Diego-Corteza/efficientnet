import torch
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Callable

class Training(torch.nn.Module):

    """
    The class 'train_model', which inherits from the class 'Model', receives four arguments:

    - model:  A model (torch.nn.Module).
    - x_data: Samples (torch.Tensor).
    - y_data: Targets of the samples (torch.Tensor).
    - lr:     Learning rate (default = 0.001).
    - epochs: Number of epochs to train (default = 100).

    And contains six methods:

    - train
    - feedforward
    - backprop
    - classify
    - evaluate_performance
    - split_data
    
    """

    lr=0.001
    epochs=100

    MAX_EPOCHS = 1000

    # def __init__(self, model, x_data, y_data, lr=None, epochs=None):
    def __init__(self, training_definition: Dict):

        super(Training, self).__init__()

        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        if lr:
            self.lr = lr
        if epochs:
            self.epochs = epochs

        self.__training_dataloader = self.create_train_dataloader()

        self.__preprocessor: [torch.nn.Module, None] = None

        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def process_yaml(self):
        # get optimzier and its parameters
        # epochs, lr
        # criterion to use



    def __call__(self, *args, **kwargs):
        for epoch in range(self.epochs):
            self.train(*args, **kwargs)
            self.validate(*args, **kwargs)
            self.test(*args...)
            if epoch >= self.MAX_EPOCHS:
                break
        self.finalize()
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
    def create_train_dataloader( yaml_data_def: dict):
        augmentation_list: List = yaml_data_def["train_augmentations"]
        return DiegosDataloader(augmentation_list)

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

    def train(self):
        """
        Method to train a model given a dataset.
        """

        self.__model.train()

        running_loss = 0.0
        for data in self.__training_dataloader:
            images, gts = data

            images.to(self.__devide)
            gts.to(self.__device)

            x = self._preprocess(x)

            preds = self.__model(images)

            # loss
            train_loss = self.criterion(preds, gts)
            train_loss.backward()

            # updates
            self.optimizer.step()

            running_loss += train_loss.item()

            # loss logger
            self.log_metric("train_loss", train_loss)

        return averaged_epoch_loss


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
