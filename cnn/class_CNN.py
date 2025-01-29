# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class model_CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
#         self.pool1 = nn.MaxPool1d(2, stride=1)
#         self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
#         self.pool2 = nn.MaxPool1d(2, stride=1)
#         self.fc1 = nn.Linear(32 * 11, 64)
#         self.fc2 = nn.Linear(64, 2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.softmax(x, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
class model_CNN(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2, stride=1)

        # Compute the output size dynamically
        dummy_input = torch.zeros(1, 1, input_length)  # Batch size = 1, 1 channel, input length
        dummy_output = self._get_conv_output(dummy_input)
        self.fc1 = nn.Linear(dummy_output, 64)
        self.fc2 = nn.Linear(64, 2)

    def _get_conv_output(self, x):
        """ Helper function to calculate output shape after conv and pooling layers. """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1).shape[1]  # Get the final flattened size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)