import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

IMAGE_WIDTH = 28

BATCH_SIZE = 10

# transforms each PIL.Image to a tensor that can be used as input in pytorch
transformations = transforms.Compose([transforms.ToTensor()])


def get_train_loader():
    train_dataset = get_dataset()
    train_loader = get_loader(train_dataset)
    return train_loader


def get_extended_train_loader():
    train_dataset = get_extended_dataset()
    train_loader = get_loader(train_dataset)
    return train_loader


def get_test_loader():
    test_dataset = get_dataset(train=False)
    test_loader = get_loader(test_dataset, shuffle=False)
    return test_loader


def train_network(model, data_loader, num_epochs, loss_function, optimizer):
    for epoch in range(num_epochs):
        model = model.train()
        for batch in enumerate(data_loader):
            i, (images, expected_outputs) = batch

            outputs = model(images)
            loss = loss_function(outputs, expected_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i == 0) or ((i+1) % (len(data_loader) / 10) == 0):
            #     p = [epoch+1, num_epochs, i+1, num_batches, loss.item()]
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(*p))

        epoch_info = "Epoch {}/{}".format(epoch+1, num_epochs)
        test_loader = get_test_loader()
        test_network(model, test_loader, epoch_info)


def test_network(model, data_loader, epoch_info=""):
    model = model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in data_loader:
            images, expected_outputs = batch

            outputs = model(images)

            # get the predicted value from each output in the batch
            predicted_outputs = torch.argmax(outputs, dim=1)

            total += expected_outputs.size(0)
            correct += (predicted_outputs == expected_outputs).sum()

        results_str = f"Test data results: {float(correct)/total}"
        if epoch_info:
            results_str += f", {epoch_info}"
        print(results_str)


def get_dataset(root="./data", train=True, transform=transformations,
                download=True):
    return datasets.MNIST(root=root, train=train, transform=transform,
                          download=download)


def get_extended_dataset(root="./data", train=True, transform=transformations,
                         download=True):
    training_dataset = datasets.MNIST(root=root, train=train,
                                      transform=transform, download=download)
    shift_operations = [identity, shift_right, shift_left, shift_up, shift_down]
    extended_dataset = []
    for image, expected_value in training_dataset:
        for shift in shift_operations:
            shifted_image = shift(image[0]).unsqueeze(0)
            extended_dataset.append((shifted_image, expected_value))
    return extended_dataset


def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle)


def identity(tensor):
    return tensor


def shift_right(tensor):
    shifted = torch.roll(tensor, 1, 1)
    shifted[:, 0] = 0.0
    return shifted


def shift_left(tensor):
    shifted = torch.roll(tensor, -1, 1)
    shifted[:, IMAGE_WIDTH-1] = 0.0
    return shifted


def shift_up(tensor):
    shifted = torch.roll(tensor, -1, 0)
    shifted[IMAGE_WIDTH-1, :] = 0.0
    return shifted


def shift_down(tensor):
    shifted = torch.roll(tensor, 1, 0)
    shifted[0, :] = 0.0
    return shifted
