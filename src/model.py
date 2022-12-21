import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(
            self, input_size=768, number_params=7,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, number_params),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
DEVICE = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# class Net(nn.Module):
#     def __init__(self, input_size=768, number_params=1):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 24, 7)
#         self.conv2 = nn.Conv2d(24, 56, 7)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(168, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 5)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x