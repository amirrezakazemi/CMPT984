import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Fully Connected class
class FC(nn.Module):
    def __init__(self, input_dim, layers_dim, output_dim):
        super(FC, self).__init__()

        self.type_str = 'FC'
        self.dropout = nn.Dropout(p=0.2)  # dropout layer
        self.FC = None

        ### Stacking layers
        layers = list()
        all_layers_dim = layers_dim
        all_layers_dim.insert(0, input_dim)
        for i in range(len(all_layers_dim) - 1):
            layers.append(nn.Linear(all_layers_dim[i], all_layers_dim[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(all_layers_dim[-1], output_dim))
        self.FC = nn.Sequential(*layers)

    def forward(self, x):
        # computing y_hat
        dropped_out = self.dropout(x)
        y_hat = self.FC(dropped_out)
        return y_hat

    def get_loss(self, y, y_hat):
        # Mean Squared Error
        return F.mse_loss(y, y_hat)

