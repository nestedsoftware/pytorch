import torch
import torch.nn as nn

from common import (get_train_loader, get_test_loader, train_network,
                    test_network, IMAGE_WIDTH)

INPUT_SIZE = IMAGE_WIDTH * IMAGE_WIDTH
OUTPUT_SIZE = 10
NUM_EPOCHS = 30
LEARNING_RATE = 3.0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(INPUT_SIZE, 100)
        self.output_layer = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        x = torch.sigmoid(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


def expand_expected_output(tensor_of_expected_outputs, output_size):
    return torch.tensor([expand_single_output(expected_output.item(),
                                              output_size)
                         for expected_output in tensor_of_expected_outputs])


# Expand expected output for comparison with the outputs from the network,
# e.g. convert 3 to [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
def expand_single_output(expected_output, output_size):
    x = [0.0 for _ in range(output_size)]
    x[expected_output] = 1.0
    return x


def create_loss_function(loss_function, output_size=OUTPUT_SIZE):
    def calc_loss(outputs, target):
        targets = expand_expected_output(target, output_size)
        return loss_function(outputs, targets)
    return calc_loss


def run_network(net):
    train_loader = get_train_loader()
    mse_loss_function = nn.MSELoss()
    loss_function = create_loss_function(mse_loss_function)
    sgd = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    train_network(net, train_loader, NUM_EPOCHS, loss_function, sgd)

    print("")

    test_loader = get_test_loader()
    test_network(net, test_loader)


if __name__ == "__main__":
    network = Net()
    run_network(network)
