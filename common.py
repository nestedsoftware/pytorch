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