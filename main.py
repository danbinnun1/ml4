import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, train_x_file, train_y_file, transform=None, target_transform=None):
        lables = list(np.loadtxt(train_y_file, dtype='long'))
        self.lables = list(np.loadtxt(train_y_file, dtype='long'))[:int(len(lables) * 0.8)]
        self.images = torch.from_numpy(np.loadtxt(train_x_file, dtype='float32'))[:int(len(lables) * 0.8)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, item):
        return self.images[item], self.lables[item]


class ValidationImageDataset(Dataset):
    def __init__(self, train_x_file, train_y_file, transform=None, target_transform=None):
        lables = list(np.loadtxt(train_y_file, dtype='long'))
        self.lables = list(np.loadtxt(train_y_file, dtype='long'))[int(len(lables) * 0.8):]
        self.images = torch.from_numpy(np.loadtxt(train_x_file, dtype='float32'))[int(len(lables) * 0.8):]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, item):
        return self.images[item], self.lables[item]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sizes = [784, 100, 50, 10]
num_epochs = 5000
batch_size = 100
learning_rate = 0.001

train_data = CustomImageDataset('./train_x', './train_y')
validation_data = ValidationImageDataset('./train_x', './train_y')
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
# validation_dataloader=DataLoader(validation_data,batch_size=64,shuffle=False)
test_data = torchvision.datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=False)

# train_features, train_labels = next(iter(train_dataloader))
#
#
# m=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
#
# train_dataloader=DataLoader(m,batch_size=64,shuffle=True)
# a,b =next(iter(train_dataloader))
# y=0

class NeuralNet(nn.Module):
    def __init__(self, layers_sizes):
        super(NeuralNet, self).__init__()
        self.sizes = layers_sizes
        self.l1 = nn.Linear(784, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, 10)
        self.layers = [self.l1, self.l2, self.l3]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.layers[-1](x)
        return x


model = NeuralNet(layers_sizes=sizes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i, (images, lables) in enumerate(train_dataloader):
        images = images.to(device)
        labels = lables.to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

    loss = criterion(model(train_dataloader.dataset.images),
                     torch.Tensor(train_dataloader.dataset.lables).to(torch.int64))
    print(f'train: epoch {epoch + 1} /{num_epochs}, loss={loss.item():.4f}')
    loss = criterion(model(validation_data.images),
                     torch.Tensor(validation_data.lables).to(torch.int64))
    print(f'validation: epoch {epoch + 1} /{num_epochs}, loss={loss.item():.4f}')
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for images,lables in test_dataloader:
            images=images.reshape(-1,784).to(device)
            lables=lables.to(device)
            outputs=model(images)

            _,predictions=torch.max(outputs,1)
            n_samples+=lables.shape[0]
            n_correct+=(predictions==lables).sum().item()
        acc = 100.0*n_correct/n_samples
        print(f'accuracy = {acc}')
