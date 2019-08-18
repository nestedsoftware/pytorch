import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

BATCH_SIZE = 10

# transforms each PIL.Image to a tensor that can be used as input in pytorch
transformations = transforms.Compose([transforms.ToTensor()])


def get_dataset(root="./data", train=True, transform=transformations,
                download=True):
    return datasets.MNIST(root=root, train=train, transform=transform,
                          download=download)


def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle)


def get_train_loader():
    train_dataset = get_dataset()
    train_loader = get_loader(train_dataset)
    return train_loader


def get_test_loader():
    test_dataset = get_dataset(train=False)
    test_loader = get_loader(test_dataset, shuffle=False)
    return test_loader


def train_network(data_loader, model, num_epochs, optimizer, reshape_input,
                  calc_loss):
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


def test_network(data_loader, model, reshape):
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in data_loader:
            images, expected_outputs = batch

            images = reshape(images)
            outputs = model(images)

            # get the predicted value from each output in the batch
            predicted_outputs = torch.argmax(outputs, dim=1)

            total += expected_outputs.size(0)
            correct += (predicted_outputs == expected_outputs).sum()

        print(f"Test data results: {float(correct)/total}")
