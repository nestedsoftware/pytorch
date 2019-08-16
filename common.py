import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

BATCH_SIZE = 10

# transforms each PIL.Image to a tensor that can be used as input in pytorch
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform,
                              download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


def train(data_loader, model, num_epochs, optimizer, reshape_input, calc_loss):
    num_batches = len(data_loader)
    for epoch in range(num_epochs):
        for batch in enumerate(data_loader):
            i, (images, expected_outputs) = batch

            images = reshape_input(images)
            outputs = model(images)
            loss = calc_loss(outputs, expected_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0 or (i+1) == num_batches:
                p = [epoch+1, num_epochs, i+1, num_batches, loss.item()]
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(*p))


def test(data_loader, model, reshape):
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in data_loader:
            images, expected_outputs = batch

            images = reshape(images)
            outputs = model(images)

            # get the maximum value from each output in the batch
            predicted_outputs = torch.max(outputs.data, 1).indices

            total += expected_outputs.size(0)
            correct += (predicted_outputs == expected_outputs).sum()

        print(f"Test data results: {float(correct)/total}")
