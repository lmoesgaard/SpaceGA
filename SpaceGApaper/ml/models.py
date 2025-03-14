import torch.nn as nn


class NN1(nn.Module):
    def __init__(self, input_dim):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)  # Output layer for regression

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad], [0.01 for p in self.parameters() if p.requires_grad]  # L2 regularization with factor 0.001