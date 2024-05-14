import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms


class ImageFolder:
    def __init__(self):
        self.data: DataLoader | None = None
        self.testing_data: DataLoader | None = None
        self.root: str | None = None
        self.image_width: int | None = None
        self.image_height: int | None = None
        self._batch_size: int | None = None
        self.split_factor: float | None = None
        self._standardize: bool = False

    def image_folder(self, root, image_width, image_height):
        self.root = root
        self.image_width = image_width
        self.image_height = image_height

    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def split(self, factor):
        self.split_factor = factor

    def standardize(self, standardize=True):
        self._standardize = standardize

    def set_data(self):
        dataset = torchvision.datasets.ImageFolder(root=self.root)

        if self._standardize:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
            dataset.transform = transform

        if self.split_factor is not None:
            data_length = len(DataLoader(dataset))

            testing_data_length = int(data_length * self.split_factor)
            training_data_length = data_length - testing_data_length

            training_subset, testing_subset = random_split(dataset, (training_data_length, testing_data_length))

            training_data_loader = DataLoader(training_subset, shuffle=True)
            testing_data_loader = DataLoader(testing_subset, shuffle=True)

            if self._batch_size is not None:
                training_data_loader.batch_size = self._batch_size
                testing_data_loader.batch_size = self._batch_size

            self.data = training_data_loader
            self.testing_data = testing_data_loader

        else:
            data_loader = DataLoader(dataset, shuffle=True)

            if self._batch_size is not None:
                data_loader.batch_size = self._batch_size

            self.data = data_loader


class Trainer:
    def __init__(self):
        self._data: ImageFolder | None = None
        self._model: torch.nn.Module | None = None
        self._criterion = None
        self._epochs: int | None = None
        self._learning_rate: float | None = None
        self._momentum: float | None = None
        self._optimize: bool = False
        self._graph: bool = False

    def data(self, data):
        self._data = data

    def model(self, model):
        self._model = model

    def criterion(self, criterion):
        self._criterion = criterion

    def epochs(self, epochs):
        self._epochs = epochs

    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def momentum(self, momentum):
        self._momentum = momentum

    def optimize(self, optimize=True):
        self._optimize = optimize

    def graph(self, graph=True):
        self._graph = graph

    def train(self):
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._learning_rate,
            momentum=self._momentum
        )

        for epoch in range(self._epochs):
            self._model.train()

            for inputs, outputs in self._data.data:
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                predictions = self._model(inputs)
                loss = self._criterion(predictions, outputs)

                loss.backward()

                if self._optimize:
                    optimizer.step()
                    optimizer.zero_grad()

                print(f'Loss: {loss}')

            self._model.eval()

            if self._data.testing_data:
                pass

            loss = 0
            correct = 0

            with torch.no_grad():
                for inputs, outputs in self._data.testing_data:
                    inputs = inputs.to(device)
                    outputs = outputs.to(device)

                    predictions = self._model(inputs)

                    loss += self._criterion(predictions, outputs).item()

                    for prediction, label in zip(predictions, outputs):
                        if torch.argmax(prediction).item() == label.item():
                            correct += 1

            print(f'Loss: {loss}, Correct: {correct}')

    def save(self, path):
        torch.jit.script(self._model).save(path)


if __name__ == '__main__':
    IMAGE_SIZE = 100

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

    data = ImageFolder()

    data.image_folder('images', 100, 100)
    data.split(0.8)
    data.batch_size(32)
    data.standardize()

    data.set_data()

    trainer = Trainer()

    trainer.data(data)
    trainer.model(None)
    trainer.criterion(torch.nn.NLLLoss())
    trainer.epochs(128)
    trainer.learning_rate(1e-3)
    trainer.momentum(0.9)
    trainer.optimize()
    trainer.graph()

    trainer.train()
    trainer.save('models')
