import torch
import torch.nn as nn

from common import (get_train_loader, get_test_loader, train_network,
                    test_network)

INPUT_SIZE = 28*28
OUTPUT_SIZE = 10
NUM_EPOCHS = 60


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.fc1 = nn.Linear(4*4*40, 100)
        self.fc2 = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2, 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2, 2))
        x = x.view(-1, 4*4*40)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # CrossEntropyLoss applies its own log softmax
        return x


def create_input_reshaper():
    def reshape(images):
        return images  # not required for convolutional layer
    return reshape


if __name__ == "__main__":
    net = Net()
    loss_func = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(net.parameters(), lr=0.03, weight_decay=0.00001)

    train_loader = get_train_loader()
    train_network(train_loader, net, NUM_EPOCHS, sgd, create_input_reshaper(),
                  loss_func)

    print("")

    test_loader = get_test_loader()
    test_network(test_loader, net, create_input_reshaper())
