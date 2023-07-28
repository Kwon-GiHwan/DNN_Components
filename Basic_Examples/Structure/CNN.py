import torch
from torch import nn

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_fo = nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2)
        self.relu_fo = nn.ReLU()
        self.maxpool_fo = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_re = nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2)
        self.relu_re = nn.ReLU()
        self.maxpool_re = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.conv_fo(x)
        out = self.relu_fo(out)
        out = self.maxpool_fo(out)


        out = self.conv_re(out)
        out = self.relu_re(out)
        out = self.maxpool_re(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out