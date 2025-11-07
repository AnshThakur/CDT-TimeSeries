import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random


# ================ standard RNN and LSTM classifiers ================
class RNNClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNNCell(input_dim, hidden_dim)  # RNN Cell
        self.fc = nn.Linear(hidden_dim, output_dim)  # fully connected layer: maps last hidden vector to model prediction
        self.device = device


    def forward(self, x):
        x = x.to(self.fc.weight.dtype).to(self.device)
        hidden = self.init_hidden(x)
        
        time_steps = x.shape[1]                 # shape of x is (batches, time_steps, features)
        for i in range(0, time_steps):
            inputs = x[:,i]                     # shape of x is (batch, features) 
            hidden = self.rnn(inputs, hidden)
            
        out = self.fc(hidden)
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=self.device)
        return h0


class LSTMClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)  # LSTM cell
        self.fc = nn.Linear(hidden_dim, output_dim)   # fully connected layer: maps last hidden vector to model prediction
        self.device = device


    def forward(self, x):
        x = x.to(self.fc.weight.dtype).to(self.device)
        hidden, cell = self.init_hidden(x)
        
        time_steps = x.shape[1]              # shape of x is (batches, time_steps, features)
        for i in range(0, time_steps):
            inputs = x[:,i]                  # shape of inputs is (batch, features)
            hidden, cell = self.rnn(inputs, (hidden,cell))
        
        out = self.fc(hidden)            
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=self.device)
        c0 = torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=self.device)
        return h0, c0
# ================ standard RNN and LSTM classifiers ================    
# ========================================
# get_net(): builds LSTM or RNN model
# If not specified (or invalid), randomly picks one
# Hidden dim randomly chosen from {32, 64, 128}
# ========================================



def get_net(net_type: str = None,
            input_dim: int = None,
            device: torch.device = None,
            output_dim: int = 1,
            hidden_dim: int = None):
    """
    Returns an LSTM or RNN model for the MIMIC or time-series dataset.

    - If `net_type` is not provided or invalid, it is randomly chosen from {'lstm', 'rnn'}.
    - If `hidden_dim` is not provided, it is randomly chosen from {32, 64, 128}.

    Args:
        net_type (str, optional): "lstm" or "rnn". Randomly chosen if None or invalid.
        input_dim (int): Number of input features.
        device (torch.device): Device to move model to.
        output_dim (int, optional): Output dimension (default=1).
        hidden_dim (int, optional): Hidden dimension. Randomly chosen if None.

    Returns:
        model (torch.nn.Module): The initialized model moved to the specified device.
    """

    # --- Validate / assign model type ---
    valid_types = ["lstm", "rnn"]
    if net_type is None or net_type.lower() not in valid_types:
        net_type = random.choice(valid_types)

    # --- Handle hidden dimension ---
    if hidden_dim is None:
        hidden_dim = random.choice([32, 64, 128])

    # --- Create model ---
    if net_type.lower() == "lstm":
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
            output_dim=output_dim
        )
    else:  # net_type == "rnn"
        model = RNNClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
            output_dim=output_dim
        )

    # --- Optional: print model info ---
    # print(f"Creating {net_type.upper()} model | hidden_dim = {hidden_dim}")

    return model.to(device)




