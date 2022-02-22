import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torchvision.models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(input)
        
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            pred = output.argmax(dim=1)
            correct_cnt += (pred==label).sum().item()
            total_cnt += label.shape[0]

    print("Test dataset accuracy: {}/{} {:.2f}".format(correct_cnt, total_cnt, correct_cnt / total_cnt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--test-batchsize", "-tb", type=int, default=100)
    parser.add_argument("--epochs", "-e", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--cuda", action='store_true', default=False)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print("Used device is {}".format(device))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batchsize, shuffle=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        lr_scheduler.step()


if __name__ == "__main__":
    main()



        


