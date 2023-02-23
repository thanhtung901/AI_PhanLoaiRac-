import time
import numpy as np
from torch import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
import copy
resize = 224
batch_size = 16
path = './trained29.pth'
path_data = './dataset/test/'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
}
dataset = torchvision.datasets.ImageFolder(path_data, transform=data_transforms['train'])

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 6)

model = net
model.load_state_dict(torch.load(path))
model.eval()


criterior = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0., momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

acc = []
loss = []
def test(model, test_loader):
    # model = model.to(device)
    running_loss = 0.0
    for images, labels in test_loader:
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = model(images)
        test_loss = criterior(outputs, labels).detach().cpu().numpy()
        running_loss += (test_loss.item() * images.size(0))

        test_predictions = torch.argmax(outputs, dim=1)
        test_accuracy = torch.mean((test_predictions == labels).float()).detach().cpu().numpy()
        acc.append(test_accuracy)
        loss.append(test_loss)
        # print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_accuracy:.3f}")
    mean = sum(acc) / len(acc)
    print(mean)

def showplt():

    plt.title("test")

    plt.plot(acc,label="acc",color = 'r')
    plt.plot(loss,label="loss", color = 'g')

    plt.xlabel("images")
    plt.ylabel("values")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    test(model, test_loader)
    showplt()



