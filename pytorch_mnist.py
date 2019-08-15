import torch
import torch.nn as nn

from common import train_loader, test_loader

INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10
NUM_EPOCHS = 30
LEARNING_RATE = 3.0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(INPUT_SIZE, 30)
        self.output_layer = nn.Linear(30, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.sigmoid(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


net = Net()


def expand_expected_outputs(tensor_of_predicted_outputs):
    return torch.tensor([expand(predicted_output.item())
                         for predicted_output in tensor_of_predicted_outputs])


def expand(predicted_output):
    x = [0.0 for _ in range(OUTPUT_SIZE)]
    x[predicted_output] = 1.0
    return x

# Remove channel and flatten images from 2-d to 1-d, i.e.
# in this, case convert the tensor from size (10, 1, 28, 28)
# to size (10, 784).
# The first argument, `-1` is a placeholder whose value
# is derived such that the total number of scalar values in
# the tensor does not change.
# Since we specify `input_size`, which is 28*28, or 784,
# the first argument will become the number of batches,
# which is 10 in this case.
def reshape_inputs(inputs, input_size):
    return inputs.view(-1, input_size)

def train(data_loader, model, epochs, input_size, calc_loss, sgd, reshape):
    num_batches = len(data_loader)
    for epoch in range(epochs):
        for batch in enumerate(data_loader):
            i, (images, expected_outputs) = batch

            images = reshape(images, input_size)

            outputs = model(images)
            loss = calc_loss(outputs, expand_expected_outputs(expected_outputs))

            optimizer.zero_grad()
            loss.backward()
            sgd.step()

            if (i+1) % 100 == 0 or (i+1) == num_batches:
                p = [epoch+1, epochs, i+1, num_batches, loss.item()]
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(*p))


def test(data_loader, model, input_size, reshape):
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in data_loader:
            images, expected_outputs = batch

            images = reshape(images, input_size)
            outputs = model(images)

            # get the maximum value from each output in the batch
            predicted_outputs = torch.max(outputs.data, 1).indices

            total += expected_outputs.size(0)
            correct += (predicted_outputs == expected_outputs).sum()

        print(f"Test data results: {float(correct)/total}")


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
train(train_loader, net, NUM_EPOCHS, INPUT_SIZE, criterion, optimizer,
      reshape_inputs)

print("")

test(test_loader, net, INPUT_SIZE, reshape_inputs)
