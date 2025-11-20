
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class MarkedDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, is_trigger=0, allow_empty=True):
        super().__init__(root, transform=transform, allow_empty=allow_empty)
        self.mark = is_trigger

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.mark

class ExtractDataset(Dataset):
    def __init__(self, source_dataset_notrans, source_dataset, source_model, transform=None):
        self.source_dataset = source_dataset
        self.source_dataset_notrans = source_dataset_notrans
        self.source_model = source_model
        self.transform = transform
        self.targets = []
        self.batch_size = 64
        self.__get_predictions__()

    def __len__(self):
        return len(self.source_dataset)
    
    def __getitem__(self, index):
        if self.transform is None:
            return self.source_dataset_notrans[index][0], self.targets[index]
        return self.transform(self.source_dataset_notrans[index][0]), self.targets[index]

    def __get_predictions__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.source_model.eval()
        self.source_model.to(device)
        dataloader = DataLoader(self.source_dataset, self.batch_size)
        for x, y in tqdm(dataloader, desc='extracting data'):
            x = x.to(device)
            outputs = self.source_model(x)
            preds = torch.argmax(outputs, 1)
            self.targets.extend(preds.cpu().numpy())