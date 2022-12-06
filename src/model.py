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
