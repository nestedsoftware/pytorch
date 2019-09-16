import os

from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

from torchvision.datasets.utils import makedir_exist_ok

IMAGE_WIDTH = 28

BATCH_SIZE = 10

# Transforms each PIL.Image to a tensor that can be used as input in pytorch
# For source of normalization values, see data_normalization_calculations.md
# We use the same normalization for the test data as what was used for training
normalization = transforms.Normalize((0.1305,), (0.3081,))
transformations = transforms.Compose([transforms.ToTensor(), normalization])


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


def get_extended_dataset(root="./data", transform=transformations,
                         download=True):
    return ExtendedMNISTDataSet(root=root, transform=transform,
                                download=download)


def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle)


# ExtendedMNISTDataSet is designed to match the logic used for MNIST dataset in
# https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html
class ExtendedMNISTDataSet(Dataset):
    def __init__(self, root, transform, download):
        self.root = root
        self.transform = transform
        self.download = download

        self.training_file = 'training.pt'
        self.training_dir_path = os.path.join(self.root,
                                              self.__class__.__name__)
        self.training_file_path = os.path.join(self.training_dir_path,
                                               self.training_file)

        if not os.path.exists(self.training_file_path):
            print("generating extended training data...")
            makedir_exist_ok(self.training_dir_path)

            self.data, self.targets = self.generate_extended_data()

            with open(self.training_file_path, 'wb') as f:
                torch.save((self.data, self.targets), f)
        else:
            print("loading extended training data from file...")
            self.data, self.targets = torch.load(self.training_file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], int(self.targets[index])
        image = Image.fromarray(image.numpy(), mode='L')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def generate_extended_data(self):
        training_dataset = datasets.MNIST(root=self.root, train=True,
                                          transform=self.transform,
                                          download=self.download)

        shift_operations = [identity, shift_right, shift_left, shift_up,
                            shift_down]
        extended_images = [shift(image)
                           for image in training_dataset.data
                           for shift in shift_operations]
        extended_targets = [target for target in training_dataset.targets
                            for _ in shift_operations]

        return extended_images, extended_targets


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
