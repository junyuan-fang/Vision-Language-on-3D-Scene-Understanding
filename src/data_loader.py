import h5py
from torch.utils.data import Dataset
import torch


class HDF5Dataset(Dataset):
    """
    h5_file='path/to/your_dataset.h5', 
    transform=your_transform_function_from_clip's_preprocess
    """
    def __init__(self, h5_file, transform=None, tokenization = None, prompt = None):
        self.h5_file = h5_file
        self.transform = transform
        self.tokenization = tokenization
        self.prompt = prompt
        
        with h5py.File(self.h5_file, 'r') as file:
            self.length = len(file['label'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            data = file['data'][idx]
            label = file['label'][idx]#
            category = file['category'][label]#string

        if self.transform:
            data = self.transform(data)
            label = torch.tensor(label)
        
        if self.tokenization and self.prompt:
            category = self.tokenization(self.prompt.replace("*",category))

        return data, category
    def get_categories(self):
        with h5py.File(self.h5_file, 'r') as file:
            categories = file['category'][:]
            categories = [category.decode('utf-8') for category in categories]
        return categories