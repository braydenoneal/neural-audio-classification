import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.backends import mps
from torch import nn
import torchvision
from torchvision import transforms

IMAGE_SIZE = 100
MOMENTUM = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 512

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

training_data, testing_data = random_split(
    torchvision.datasets.ImageFolder(root='images', transform=transform), (24_000, 6_000)
)

training_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
testing_data_loader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'mps' if mps.is_available() else 'cpu'


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(50 * (IMAGE_SIZE // 2 // 2) ** 2, IMAGE_SIZE * IMAGE_SIZE),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, forward_xss):
        return self.network(forward_xss)


model = ConvolutionalModel().to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def train(data_loader, model, loss_fn, optimizer):
    model.train()

    size = len(data_loader.dataset)

    for batch, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'loss: {loss.item():>7f}  [{(batch + 1) * len(images):>5d}/{size}]')


def test(data_loader, model, loss_fn):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            test_loss += loss_fn(predictions, labels).item()

            for prediction, label in zip(predictions, labels):
                if torch.argmax(prediction).item() == label.item():
                    correct += 1

    test_loss /= len(data_loader)
    correct /= len(data_loader.dataset)

    print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


print(f'Training on {device}\n')

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    train(training_data_loader, model, criterion, optimizer)
    test(testing_data_loader, model, criterion)

print('Done!')

torch.jit.script(model).save('models/07.pt')

"""
# Torch-Light library example
import torchlight

data = torchlight.data.ImageFolder(
    folder_path='images',
    train_test_split=0.8,
)

trainer = torchlight.Trainer(data)

trainer.options.learningRate(1e-3)
trainer.options.momentum(0.9)
trainer.options.optimize()
trainer.options.criterion(torchlight.NLLLoss())
trainer.options.standardize()
trainer.options.graph()

trainer.train()
trainer.save('models')
"""

"""
import torchlight

data = torchlight.Data()
data.image_folder('images')
data.split(0.8)

trainer = torchlight.Trainer()

trainer.data = data
trainer.learning_rate = 1e-3
trainer.momentum = 0.9
trainer.optimize = True
trainer.criterion = torchlight.nll_loss
trainer.standardize = True
trainer.graph = True

trainer.train()
trainer.save('models')
"""

"""
import torchlight

data = torchlight.Data(
    image_folder='images',
    split=True
    split_amount=0.8
)

torchlight.train(
    data=data,
    learning_rate=1e-3,
    momentum=0.9,
    optimize=True,
    criterion=torchlight.nll_loss(),
    normalize=True,
    center=True,
    graph=True,
    save=True,
    save_path='models'
)
"""

"""
import torchlight

data = torchlight.Data()

data.image_folder('models')
data.split(factor=0.8)
data.center()
data.normalize()

trainer = torchlight.Trainer()

trainer.options.data(data)
trainer.options.model(model)
trainer.options.criterion(torchlight.nll_loss())
trainer.options.epochs(128)
trainer.options.batch_size(32)
trainer.options.learning_rate(1e-3)
trainer.options.momentum(0.9)
trainer.options.optimize()
trainer.options.graph()

trainer.train()
trainer.save('models')
"""
