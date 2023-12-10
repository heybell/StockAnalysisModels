import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()

        self.fc = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        return self.fc(x)