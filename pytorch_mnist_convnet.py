import torch
import torch.nn as nn

from common import (get_train_loader, get_test_loader,
                    train_network, test_network)

OUTPUT_SIZE = 10


class ConvNetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.fc1 = nn.Linear(12*12*20, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 12*12*20)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.out(x)
        return x


class ConvNetTwoConvLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(4*4*40, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 4*4*40)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.out(x)
        return x


class ConvNetTwoConvLayersReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(4*4*40, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 4*4*40)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x


class ConvNetTwoConvTwoDenseLayersWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4*4*40, 1000)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)

        self.dropout3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(1000, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 4*4*40)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.dropout3(x)
        x = self.out(x)
        return x


def train_and_test_network(net, num_epochs=60, lr=0.1, wd=0,
                           loss_function=nn.CrossEntropyLoss(),
                           train_loader=get_train_loader(),
                           test_loader=get_test_loader()):
    sgd = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    train_network(net, train_loader, num_epochs, loss_function, sgd)

    print("")

    test_network(net, test_loader)


if __name__ == "__main__":
    train_and_test_network(ConvNetSimple())
