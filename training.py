import torch
from sklearn.model_selection import train_test_split

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

    def __init__(self, model, x_data, y_data, lr=None, epochs=None):

        super(Training, self).__init__()

        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        if lr:
            self.lr = lr
        if epochs:
            self.epochs = epochs


    def train(self):
        """
        Method to train a model given a dataset.
        """


    def feedforward(self):
        """
        Method to feedforward data through the net.
        """


    def backprop(self):
        """
        Method to backpropagate data through the net.
        """


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