import torch
import torchvision
from torchvision import transforms


def load_mnist(path='./data', download=True, batch_size=100, shift_pixels=2):
    """
    Construct dataloaders for training and test data. Data augmentation is also done here.
    :param path: file path of the dataset
    :param download: whether to download the original data
    :param batch_size: batch size
    :param shift_pixels: maximum number of pixels to shift in each direction
    :return: train_loader, test_loader
    """
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_data = torchvision.datasets.MNIST(path, train=True, download=download,
                                            transform=transforms.Compose(
                                                [transforms.RandomCrop(size=28, padding=shift_pixels),
                                                 transforms.ToTensor()]))
    test_data = torchvision.datasets.MNIST(path, train=False, download=download,transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader
