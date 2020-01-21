import torch.nn as nn
import torchvision.models
from skorch import NeuralNetClassifier
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np


class NewResNet(nn.Module):

    def __init__(self, out_size=2, freeze=False, pretrained=True, arch='resnet50'):
        """
        This method initializes a resnet model but appends a layer to the beginning of the model.
        :param out_size: Number of errors to train
        :param freeze: Boolean indicating to freeze pretrained layers
        :param pretrained: Boolean indicating the use of pretrained network
        :param arch: The model architecture to use
        """

        super().__init__()

        if arch == 'resnet50':
            net = torchvision.models.resnet50(pretrained=pretrained)
            self.model_name = 'resnet50'
        elif arch == 'resnet18':
            net = torchvision.models.resnet18(pretrained=pretrained)
            self.model_name = 'resnet18'
        elif arch == 'resnet34':
            net = torchvision.models.resnet34(pretrained=pretrained)
            self.model_name = 'resnet34'
        elif arch == 'resnet101':
            net = torchvision.models.resnet101(pretrained=pretrained)
            self.model_name = 'resnet101'
        elif arch == 'resnet152':
            net = torchvision.models.resnet152(pretrained=pretrained)
            self.model_name = 'resnet152'
        elif arch == 'wide_resnet50_2':
            net = torchvision.models.wide_resnet50_2(pretrained=pretrained)
            self.model_name = 'wide_resnet50_2'
        elif arch == 'wide_resnet101_2':
            net = torchvision.models.wide_resnet101_2(pretrained=pretrained)
            self.model_name = 'wide_resnet101_2'
        else:
            net = torchvision.models.resnet18(pretrained=pretrained)
            self.model_name = 'resnet18'

        if pretrained and freeze:
            for param in net.parameters():
                param.requires_grad = False

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_size)

        self.fcOut = net.fc
        self.pretrained_net = net


    def forward(self, x):
        """
        Method used to define how an observation gets passed through the network. First the observation
        will be passed through the new layer, perform batch normalization and then relu activation
        before it is passed to the trained network
        :param x: observation input into the network
        :return: The result of passing the observation through the network.
        """

        out = self.pretrained_net(x)

        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CNN = NeuralNetClassifier(
    NewResNet,
    max_epochs=10,
    lr=0.0001,
    device=device,
    optimizer=torch.optim.Adam
)

train_path = 'data/training'
train_data = torchvision.datasets.ImageFolder(
    root=train_path,
    transform = torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

valid_path = 'data/validation'
valid_data = torchvision.datasets.ImageFolder(
    root=valid_path,
    transform=torchvision.transforms.ToTensor()
)

valid_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

# transform = transforms.Compose([
#
#     transforms.Resize(256),
#
#     transforms.RandomCrop(224),
#
#     transforms.RandomHorizontalFlip(),
#
#     transforms.ToTensor(),
#
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# # MNIST dataset (images and labels)
#
# train_dataset = torchvision.datasets(root='data/training',
#
#                                            train=True,
#
#                                            transform=transform,
#
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='data/test',
#
#                                           train=False,
#
#                                           transform=transform,
#
#                                           download=True)
