
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim

from torch.utils.data import Dataset, DataLoader


class Data(Dataset):

    # Constructor
    def __init__(self, train=True):
        # Create x values from -3 to 3 in steps of 0.1
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        # Create f(x) = -3x + 1
        self.f = -3 * self.x + 1
        # Add noise with random values between -0.1 and 0.1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        # Length of dataset
        self.len = self.x.shape[0]

        # outliers to train with
        if train == True:
            self.y[0] = 0
            self.y[50:55] = 20
        else:
            pass

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

# Create training dataset and validation dataset

train_data = Data()
# Create validation dataset with train = False which means it will not have outliers
val_data = Data(train = False)

 # Here we plot the data with the outliers which are apparent

plt.plot(train_data.x.numpy(), train_data.y.numpy(), 'xr',label="training data ")
plt.plot(train_data.x.numpy(), train_data.f.numpy(),label="true function  ")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()