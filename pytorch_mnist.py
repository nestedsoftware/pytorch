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


def reformat_images(images, input_size):
    # Remove channel and flatten images from 2-d to 1-d, i.e.
    # assuming batch size of 10, convert the tensor from
    # size (10, 1, 28, 28) to size (10, 784).
    # The first argument, `-1` is a placeholder whose value
    # is derived such that the total number of scalar values in
    # the tensor does not change.
    # Since we specify `input_size`, which is 28*28, or 784,
    # the placeholder argument will become the number of batches,
    # which is 10 in this case.
    return images.reshape(-1, input_size)


def create_input_reshaper(input_size=INPUT_SIZE):
    def reshape(images):
        return reformat_images(images, input_size)
    return reshape


def create_loss_function(loss_function, output_size=OUTPUT_SIZE):
    def calc_loss(outputs, target):
        targets = expand_expected_output(target, output_size)
        return loss_function(outputs, targets)
    return calc_loss


def run_network(net):
    mse_loss_function = nn.MSELoss()
    sgd = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    train_loader = get_train_loader()
    train_network(train_loader, net, NUM_EPOCHS, sgd,
                  create_input_reshaper(),
                  create_loss_function(mse_loss_function))

    print("")

    test_loader = get_test_loader()
    test_network(test_loader, net, create_input_reshaper())


if __name__ == "__main__":
    network = Net()
    run_network(network)
