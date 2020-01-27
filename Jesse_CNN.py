import torch.nn as nn
import torchvision.models
from skorch import NeuralNetClassifier
import torch
import torchvision
from torchvision import transforms, datasets
from skorch.helper import predefined_split
import os.path
from skorch.callbacks import LRScheduler, Checkpoint, Freezer


class NewResNet(nn.Module):

    def __init__(self, out_size=2, freeze=False, pretrained=True, arch='resnet50'):
        """
        This method initializes a resnet model but appends a layer to the beginning of the model.
        :param out_size: Number of errorstrain_x, train_y to train
        :param freeze: Boolean indicating to freeze pretrained layers
        :param pretrained: Boolean indicating the use of pretrained network
        :param arch: The model architecture to use
        """

        super().__init__()

        if arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
            self.model_name = 'resnet50'
        elif arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=pretrained)
            self.model_name = 'resnet18'
        elif arch == 'resnet34':
            model = torchvision.models.resnet34(pretrained=pretrained)
            self.model_name = 'resnet34'
        elif arch == 'resnet101':
            model = torchvision.models.resnet101(pretrained=pretrained)
            self.model_name = 'resnet101'
        elif arch == 'resnet152':
            model = torchvision.models.resnet152(pretrained=pretrained)
            self.model_name = 'resnet152'
        elif arch == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
            self.model_name = 'wide_resnet50_2'
        elif arch == 'wide_resnet101_2':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
            self.model_name = 'wide_resnet101_2'
        else:
            model = torchvision.models.resnet18(pretrained=pretrained)
            self.model_name = 'resnet18'

        if pretrained and freeze:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_size)

        self.model = model


    def forward(self, x):
        """
        Method used to define how an observation gets passed through the network. First the observation
        will be passed through the new layer, perform batch normalization and then relu activation
        before it is passed to the trained network
        :param x: observation input into the network
        :return: The result of passing the observation through the network.
        """

        out = self.model(x)

        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = 'data'

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

train_ds = datasets.ImageFolder(
    os.path.join(data_dir, 'training'),
    train_transforms
)

valid_ds = datasets.ImageFolder(
    os.path.join(data_dir, 'validation'),
    val_transforms
)

checkpoint = Checkpoint(f_params='best_model.pt', monitor='valid_acc_best')
freezer = Freezer(lambda x: not x.startswith('model.fc'))


CNN = NeuralNetClassifier(
    NewResNet,
    max_epochs=5,
    lr=0.0001,
    criterion=nn.CrossEntropyLoss,
    device=device,
    optimizer=torch.optim.Adam,
    train_split=predefined_split(valid_ds),
    batch_size=16,
    callbacks=[checkpoint, freezer],
    iterator_train__shuffle=True,
    iterator_valid__shuffle=True
)

# y=None because y is in train_ds and because it is an image folder, skorch
# knows what to do. Here's a good tutorial:
# https://github.com/skorch-dev/skorch/blob/master/notebooks/Transfer_Learning.ipynb
CNN.fit(train_ds, y=None)
