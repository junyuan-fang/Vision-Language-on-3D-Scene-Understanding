# import os
# from clip import clip
# import torch
# from torchvision.datasets import STL10
# torch.cuda.empty_cache()
# print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-L/14@336px', device)
# print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
# # Download the dataset
# stl10_dataset = STL10(root=os.path.expanduser("./data"), download=True, split='test')
# # Prepare the prompts
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in stl10_dataset.classes]).to(device)

# # Initialize variables for accuracy calculation
# correct_predictions = 0
# total_samples = len(stl10_dataset)

# print(total_samples)
# torch.cuda.empty_cache()

# import torch

# # 原始张量
# tensor = torch.tensor([[1, 2], [3, 4]])

# # 重复整个张量
# # 第一个数字表示行的重复次数，第二个数字表示列的重复次数
# repeated_tensor = tensor.repeat(2, 1)

# print(repeated_tensor)

import torch

tensor = torch.tensor([[1, 2], [3, 4]])
repeat_factor = 2

# 在第一个维度（维度 0）上重复每个元素
repeated_tensor = tensor.repeat_interleave(repeat_factor, dim=1)

print(repeated_tensor)