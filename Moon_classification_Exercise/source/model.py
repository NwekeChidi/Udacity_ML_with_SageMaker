import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        # first layer
        hidden_1 = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_1)
        # second layer
        hidden_2 = int(hidden_dim/2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # final layer
        self.fc3 = nn.Linear(hidden_2, output_dim)
        # dropout
        self.dropout = nn.Dropout(0.2)
        # sigmoid layer
        self.sig = nn.Sigmoid()
        
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        # Computing layers with  activation functions and dropout
        # add first layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # add second layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # add final layer
        x = self.fc3(x)
        # add sigmoid layer
        x = self.sig(x)
        return x