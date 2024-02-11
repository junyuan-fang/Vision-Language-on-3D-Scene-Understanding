import h5py
from torch.utils.data import Dataset
import torch
import numpy as np  

class HDF5_n_shot(Dataset):#todo
    """
    h5_file='path/to/your_dataset.h5', 
    transform=your_transform_function_from_clip's_preprocess
    """
    def __init__(self, h5_file, n_shot, transform=None, tokenization=None, prompt=None):
        self.h5_file = h5_file
        self.transform = transform
        self.tokenization = tokenization
        self.prompt = prompt
        self.n_shot = n_shot
        
        with h5py.File(self.h5_file, 'r') as file:
            self.labels = file['label'][:]
            self.categories = [category.decode('utf-8') for category in file['category'][:]]
        #self.category_to_indices字典允许你快速地通过类别名称访问所有属于该类别的样本索引，这对于实现基于类别的采样逻辑（如在n-shot学习场景中）非常有用
        self.category_to_indices = {category: torch.where(torch.tensor(self.labels) == i)[0]
                                    for i, category in enumerate(self.categories)}

    def __len__(self):
        # 这里的长度被设定为总样本数除以n_shot，你可能需要根据实际情况调整
        return len(self.labels) // self.n_shot

    def __getitem__(self, idx):
        category = self.categories[idx % len(self.categories)]
        indices = self.category_to_indices[category]
        
        # 随机选择n_shot个样本
        chosen_indices = torch.randperm(len(indices))[:self.n_shot]
        
        data, categories = [], []
        with h5py.File(self.h5_file, 'r') as file:
            for i in chosen_indices:
                datum = file['data'][i.item()]
                if self.transform:
                    datum = self.transform(datum)
                if self.tokenization and self.prompt:
                    category_text = self.tokenization(self.prompt.replace("*", category))
                else:
                    category_text = category
                data.append(datum)
                categories.append(category_text)
        
        # Stack data for returning
        data = torch.stack(data)
        # 注意：这里我们返回了一个类别的n-shot样本
        return data, categories

class HDF5Dataset(Dataset):
    """
    h5_file='path/to/your_dataset.h5', 
    transform=your_transform_function_from_clip's_preprocess
    """
    def __init__(self, h5_file, transform=None, tokenization = None, prompt = None, split = 'train', split_ratio = 0.8, seed = 0):
        self.h5_file = h5_file
        self.transform = transform
        self.tokenization = tokenization
        self.prompt = prompt
        self.split = split
        
        with h5py.File(self.h5_file, 'r') as file:
            self.length = len(file['label'])

       # 设置随机种子以确保可重复性
        np.random.seed(seed)
        
        with h5py.File(self.h5_file, 'r') as file:
            labels = np.array(file['label'])
            # 计算分割点
            num_samples = len(labels)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            split_point = int(num_samples * split_ratio)
            
            if split == 'train':
                self.indices = indices[:split_point]
            elif split == 'test':
                self.indices = indices[split_point:]
            else:
                raise ValueError("split must be 'train' or 'test'")
                
            self.categories = [category.decode('utf-8') for category in file['category'][:]] 
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with h5py.File(self.h5_file, 'r') as file:
            data = file['data'][actual_idx]
            label = file['label'][actual_idx]#
            category = file['category'][label].decode('utf-8')#string

        if self.transform:
            data = self.transform(data)
            #label = torch.tensor(label)
        
        if self.tokenization and self.prompt:
            category = self.tokenization(self.prompt.replace("*",category))

        return data, category
    def get_categories(self):
        return self.categories