import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms


class ImageFolder:
    def __init__(self):
        self.data = list[DataLoader]
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

            self.data = [training_data_loader, testing_data_loader]

        else:
            data_loader = DataLoader(dataset, shuffle=True)

            if self._batch_size is not None:
                data_loader.batch_size = self._batch_size

            self.data = [data_loader]


class Trainer:
    def __init__(self):
        self.options = TrainerOptions()

    def train(self):
        print(self.options.epochs)

    def save(self, path):
        print(f'{self.options.data} {path}')


class TrainerOptions:
    def __init__(self):
        self._data = None
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


if __name__ == '__main__':
    data = ImageFolder()

    data.image_folder('images', 100, 100)
    data.split(0.8)
    data.batch_size(32)
    data.standardize()

    data.set_data()

    trainer = Trainer()

    trainer.options.data(None)
    trainer.options.model(None)
    trainer.options.criterion(None)
    trainer.options.epochs(128)
    trainer.options.learning_rate(1e-3)
    trainer.options.momentum(0.9)
    trainer.options.optimize()
    trainer.options.graph()

    trainer.train()
    trainer.save('models')
