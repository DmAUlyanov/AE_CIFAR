from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from pathlib import Path
import numpy as np

from datasets.dataset_type import DatasetType
import config


class Cifar10(Dataset):
    r"""
    https://www.cs.toronto.edu/~kriz/cifar.html
    This class is a wrapper over the default pytorch class for ease of use for the anomaly detection task.
    Parameter 'anomaly_class' is responsible for which class will be considered anomalous, while the rest are normal.
    Available classes:
                     'airplane'
                     'automobile'
                     'bird'
                     'cat'
                     'deer'
                     'dog'
                     'frog'
                     'horse'
                     'ship'
                     'truck'
    """
    DEFAULT_DATA_PATH = config.DATA_PATH / 'cifar10'

    def __init__(self, anomaly_class: str, dataset_type: DatasetType, data_path: Path = DEFAULT_DATA_PATH):
        self.images = []
        self.labels = []
        
        if dataset_type & DatasetType.Train:
            _dataset = CIFAR10(root=str(data_path), train=True, download=True, transform=ToTensor())
            anomaly_class_idx = _dataset.class_to_idx[anomaly_class]
            imgs = _dataset.data[np.array(_dataset.targets) != anomaly_class_idx]
            lbls = np.zeros((imgs.shape[0],), dtype=int)
            self.images.append(imgs)
            self.labels.append(lbls)
            
        if dataset_type & DatasetType.Test:
            _dataset = CIFAR10(root=str(data_path), train=False, download=True, transform=ToTensor())
            anomaly_class_idx = _dataset.class_to_idx[anomaly_class]
            self.images.append(_dataset.data)
            self.labels.append((np.array(_dataset.targets) == anomaly_class_idx).astype(int))
        
        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
