import h5py
from torch.utils.data import Dataset
import torch
import numpy as np  

# class HDF5_N_ShotDataset(Dataset):
#     """
#     A dataset class for handling HDF5 files for PyTorch models, specifically designed for N-shot learning tasks.

#     Parameters:
#     - h5_file (str): Path to the HDF5 file containing the dataset.
#     - n (int): Number of samples per class to include in the support set.
#     - k (int): Number of classes to include in each N-shot task.
#     - q (int): Number of query samples per class.
#     - transform (callable, optional): A function/transform that takes in an image sample and returns a transformed version.
#     - tokenization (callable, optional): A function for tokenizing category labels if necessary.
#     - prompt (str, optional): A prompt template to be used with tokenization.
#     - split (str, optional): Specifies if this is a 'train' or 'test' dataset split. Default is 'train'.
#     - split_ratio (float, optional): The ratio of the dataset to be used for training. Ignored if split is 'test'. Default is 0.8.
#     - seed (int, optional): Random seed for reproducibility. Default is 0.
#     """
#     def __init__(self, h5_file, n, k, q, transform=None, tokenization=None, prompt=None, split='train', split_ratio=0.8, seed=0):
#         assert 0 < split_ratio < 1, "split_ratio must be between 0 and 1"
#         assert split in ['train', 'test'], "split must be 'train' or 'test'"
        
#         self.h5_file = h5_file
#         self.n = n
#         self.k = k
#         self.q = q
#         self.transform = transform
#         self.tokenization = tokenization
#         self.prompt = prompt
#         self.split = split
#         self.rng = np.random.default_rng(seed)
        
#         # Opening the HDF5 file
#         try:
#             self.file = h5py.File(self.h5_file, 'r')
#         except Exception as e:
#             raise FileNotFoundError(f"Failed to open file {self.h5_file}: {e}")
        
#         if 'label' not in self.file or 'data' not in self.file or 'category' not in self.file:
#             raise ValueError("HDF5 file must contain 'data', 'label', and 'category' datasets.")
        
#         # Prepare indices for splitting the dataset
#         labels = np.array(self.file['label'])
#         self.categories = np.unique(labels)
#         self.category_to_indices = {cat: np.where(labels == cat)[0] for cat in self.categories}
        
#         self.indices = self.prepare_indices(split, split_ratio, labels)
        
#     def prepare_indices(self, split, split_ratio, labels):
#         """
#         Prepares indices for training and testing splits.
#         """
#         indices = np.arange(len(labels))
#         self.rng.shuffle(indices)
#         split_point = int(len(indices) * split_ratio)
        
#         if split == 'train':
#             return indices[:split_point]
#         else:  # split == 'test'
#             return indices[split_point:]
    
#     def __len__(self):
#         # This might need adjustment based on how you define an epoch with N-shot tasks
#         return len(self.indices) // (self.k * (self.n + self.q))

#     def __getitem__(self, idx):
#         """
#         Returns an N-shot task consisting of support and query sets.
#         """
#         task_indices = []
#         selected_classes = self.rng.choice(self.categories, self.k, replace=False)
        
#         for cls in selected_classes:
#             cls_indices = self.rng.choice(self.category_to_indices[cls], self.n + self.q, replace=False)
#             task_indices.extend(cls_indices)
        
#         data, labels = [], []
#         for idx in task_indices:
#             with h5py.File(self.h5_file, 'r') as file:
#                 d = file['data'][idx]
#                 l = file['label'][idx]
#             if self.transform:
#                 d = self.transform(d)
#             if self.tokenization and self.prompt:
#                 l = self.tokenization(self.prompt.replace("*", l.decode('utf-8')))
#             data.append(d)
#             labels.append(l)
        
#         # Returning the data and labels as tensors
#         return torch.stack(data), torch.tensor(labels)

#     def __del__(self):
#         if hasattr(self, 'file') and self.file is not None:
#             self.file.close()

class HDF5_N_WAY_Dataset(Dataset):
    def __init__(self, h5_file, transform=None, tokenization=None, prompt=None, split='train',
                 seed=0, num_train_classes=10):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"

        self.h5_file = h5_file
        self.transform = transform
        self.tokenization = tokenization
        self.prompt = prompt
        self.split = split
        self.file = None  # Placeholder for the HDF5 file handle
        self.rng = np.random.default_rng(seed)  # Local random number generator

        # Open the HDF5 file
        try:
            self.file = h5py.File(self.h5_file, 'r')
        except Exception as e:
            raise FileNotFoundError(f"Failed to open file {self.h5_file}: {e}")

        # Ensure necessary datasets are in the file
        if 'label' not in self.file or 'data' not in self.file or 'category' not in self.file:
            raise ValueError("HDF5 file must contain 'data', 'label', and 'category' datasets.")

        # Assuming category is a dataset of unique strings
        all_unique_labels = np.arrange(0,len(self.file['category']))#np.unique(self.file['label'][:])

        # Select labels for train or test
        if split == 'train':
            selected_labels = self.rng.choice(all_unique_labels, num_train_classes, replace=False)
        else:  # For 'test', use the remaining labels
            train_labels = self.rng.choice(all_unique_labels, num_train_classes, replace=False)
            selected_labels = np.setdiff1d(all_unique_labels, train_labels)

        # Find indices of data that belong to selected labels
        self.indices = np.where(np.isin(self.file['label'][:], selected_labels))[0]
        self.rng.shuffle(self.indices)  # Shuffle indices if necessary

        self.categories = [category.decode('utf-8') for category in self.file['category']]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            raise RuntimeError("Dataset file is not open.")

        actual_idx = self.indices[idx]
        data = self.file['data'][actual_idx]
        label = self.file['label'][actual_idx]
        category = self.file['category'][label].decode('utf-8')  # string

        if np.isnan(data).any():
            print(f"NaN detected in data at index {actual_idx}, the data will be replaced with index 0's data")
            data = self.file['data'][0]
            label = self.file['label'][0]
            category = self.file['category'][label].decode('utf-8')

        transformed_data = data
        if self.transform:
            transformed_data = self.transform(data)

        category_prompt = category
        if self.tokenization and self.prompt:
            category_prompt = self.prompt.replace("*", category)

        return transformed_data, category_prompt

    def get_categories(self):
        return self.categories

    def __del__(self):
        if self.file is not None:
            self.file.close()

class HDF5Dataset(Dataset):
    """
    A dataset class for handling HDF5 files for PyTorch models.

    Parameters:
    - h5_file (str): Path to the HDF5 file containing the dataset.
    - transform (callable, optional): A function/transform from clip that takes in a sample and returns a transformed version.
    - tokenization (callable, optional): A function for tokenizing category labels if necessary.
    - prompt (str, optional): A prompt template to be used with tokenization.
    - split (str, optional): Specifies if this is a 'train' or 'test' dataset split. Default is 'train'.
    - split_ratio (float, optional): The ratio of the dataset to be used for training. Ignored if split is 'test'. Default is 0.8.
    - validation_ratio (float, optional): The ratio of the training set to be used for validation. Default is 0.1.
    - seed (int, optional): Random seed for reproducibility. Default is 0.
    """
    def __init__(self, h5_file, transform=None, tokenization = None, prompt = None, split = 'train', split_ratio = 0.8, validation_ratio =0, seed = 0):
        assert 0 < split_ratio < 1, "split_ratio must be between 0 and 1"
        assert split in ['train', 'test', 'valid'], "split must be 'train' or 'test'"
        
        self.h5_file = h5_file
        self.transform = transform
        self.tokenization = tokenization
        self.prompt = prompt
        self.split = split
        self.file = None  # Placeholder for the HDF5 file handle
        self.rng = np.random.default_rng(seed)  # Local random number generator
        
        try:
            self.file = h5py.File(self.h5_file, 'r')
        except Exception as e:
            raise FileNotFoundError(f"Failed to open file {self.h5_file}: {e}")
        
        if 'label' not in self.file or 'data' not in self.file or 'category' not in self.file:
            raise ValueError("HDF5 file must contain 'data', 'label', and 'category' datasets.")
        
        self.length = len(self.file['label'])
        num_samples = len(self.file['label'])
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)
        split_point = int(num_samples * split_ratio)
        validation_split_point = int(split_point * (1 - validation_ratio)) #0->split_point->validation_split_point->1, #0->split_point->validation_split_point->1

        if split == 'train':
            self.indices = indices[:validation_split_point]
        elif split == 'valid': # split from train
            self.indices = indices[validation_split_point:split_point]
        else:  # split == 'test'
            self.indices = indices[split_point:]
        
        self.categories = [category.decode('utf-8') for category in self.file['category'][:]]
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            raise RuntimeError("Dataset file is not open.")
        
        actual_idx = self.indices[idx]
        data = self.file['data'][actual_idx]
        label = self.file['label'][actual_idx]
        category = self.file['category'][label].decode('utf-8')  # string

        if np.isnan(data).any():
            print(f"NaN detected in data at index {actual_idx}, the data will be replaced with index 0's data")
            data = self.file['data'][0]
            label = self.file['label'][0]
            category = self.file['category'][label].decode('utf-8')
        
        if self.transform:
            transformed_data = self.transform(data)

        if self.tokenization and self.prompt:
            category_prompt = self.prompt.replace("*", category)
        
        return transformed_data, category_prompt

    def get_categories(self):
        return self.categories
    
    def __del__(self):
        if self.file is not None:
            self.file.close()