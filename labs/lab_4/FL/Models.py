from unicodedata import bidirectional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn.parameter import Parameter



class LSTMClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, dropout_prob=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)  # LSTM Cell
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()  # For binary classification
        self.device = device
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)  # Dropout with a specified probability

    def forward(self, x):
        hidden, cell = self.init_hidden(x)
        time_steps = x.shape[1]  # shape of x is (batches, time_steps, features)
        
        for i in range(time_steps):
            inputs = x[:, i]  # (batch, features) shape
            hidden, cell = self.rnn(inputs, (hidden, cell))
            
        # Apply dropout before the final fully connected layer
        hidden = self.dropout(hidden)
        
        out = self.activation(self.fc1(hidden))  # Take the hidden vector corresponding to the last time step
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim)
        c0 = torch.zeros(x.size(0), self.hidden_dim)
        return h0.to(self.device), c0.to(self.device)


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_filters, device, dropout_prob=0.5):
        super(CNNClassifier, self).__init__()
        self.device = device

        # Define 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        
        # Pooling layer to reduce temporal dimension
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(num_filters, 1)
        
        # Activation and dropout
        self.activation = nn.Sigmoid()  # For binary classification
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Input shape: (batch_size, time_steps, features)
        # Transpose to (batch_size, features, time_steps) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply first convolutional layer and activation
        x = torch.relu(self.conv1(x))
        
        # Apply second convolutional layer, pooling, and activation
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Global average pooling across the time dimension
        x = torch.mean(x, dim=2)  # Shape: (batch_size, hidden_dim)
        
        # Apply dropout before the fully connected layer
        x = self.dropout(x)
        # print(x.shape)
        # Pass through the fully connected layer and sigmoid activation
        out = self.activation(self.fc1(x))
        
        return out









