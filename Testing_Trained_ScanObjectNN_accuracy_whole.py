import os
import sys
from clip import clip
import torch 
import h5py
import numpy as np
torch.cuda.empty_cache()

file = "voxel_all.h5"
correct_predictions = 0
total_samples = 150
batch_size = 10

DATA_PATH = 'data/ScanObjectNN/object_dataset/'
READ_PATH = 'data/ScanObjectNN/processed_object_dataset/'
topk = 5

#return a 
def data_generator(file_path, batch_size):
     with h5py.File(file_path, 'r') as f:
        total_size = f['data'].shape[0]
        for start_idx in range (0,total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            yield f['data'][start_idx: end_idx], f['label'][start_idx: end_idx]

def prediction(voxel, label):
    global correct
    # Prepare the inputs
    voxel_input = preprocess(voxel).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(voxel_input)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Pick the top 5 most similar labels for the image
    values, indices = similarity[0].topk(topk)

    # Print the result
    print(f"\n {categories[label]} 's top {topk} predictions:")
    for value, index in zip(values, indices):
        category_str = categories[index]
        print(f"{category_str:>16s}: {100 * value.item():.2f}%")
        
    predicted_class = torch.argmax(similarity).item()
    if label == predicted_class:
        correct += 1

global categories, device, model, preprocess, text_inputs, correct 

correct = 0


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
model_path = 'trained_model/best_model_0223-205657__training_ratio0.2_model_ViT-B_32_train_whole_visual_layers_lr_1e-05_weight_decay_0.2_betas_(0.9, 0.98)_eps_1e-06.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
generator  = data_generator(READ_PATH+file,batch_size)


with h5py.File(READ_PATH+file, 'r') as f:
    categories = list(f['category'])
    categories = [category.decode('utf-8') for category in categories]
# Prepare the prompts
print(categories)
print(len(categories))
text_inputs = torch.cat([clip.tokenize(f"A voxelized representation of {c}") for c in categories]).to(device)



first_iteration = True
for i in range(int(total_samples/batch_size)):
    #load voxels
    voxel_arrays, labels = next(generator)
    print(labels)
    for j in range(len(voxel_arrays)):
        voxel = voxel_arrays[j]
        label = labels[j]
        prediction(voxel, label)

print(f"Accuracy: {100 * correct / total_samples:.2f}%")
torch.cuda.empty_cache()
# import concurrent.futures
# import itertools
# from clip import clip
# import torch 
# import h5py
# import numpy as np
# import queue
# import threading

# torch.cuda.empty_cache()

# file = "voxel_all.h5"
# correct_predictions = 0
# total_samples = 150
# batch_size = 10

# DATA_PATH = 'data/ScanObjectNN/object_dataset/'
# READ_PATH = 'data/ScanObjectNN/processed_object_dataset/'
# topk = 5

# def data_generator(file_path, batch_size):
#     with h5py.File(file_path, 'r') as f:
#         total_size = f['data'].shape[0]
#         for start_idx in range(0, total_size, batch_size):
#             end_idx = min(start_idx + batch_size, total_size)
#             yield f['data'][start_idx:end_idx], f['label'][start_idx:end_idx]

# def prediction(voxel, label):
#     global correct
#     # Prepare the inputs
#     voxel_input = preprocess(voxel).unsqueeze(0).to(device)

#     # Calculate features
#     with torch.no_grad():
#         image_features = model.encode_image(voxel_input)
#         text_features = model.encode_text(text_inputs)

#         # Normalize features
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         # Calculate similarity
#         similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

#     # 释放Tensor以节约内存
#     del voxel_input, image_features, text_features

#     # Pick the top 5 most similar labels for the image
#     values, indices = similarity[0].topk(topk)

#     # Print the result
#     print(f"\n {categories[label].decode('utf-8')} 's top {topk} predictions:")
#     for value, index in zip(values, indices):
#         category_str = categories[index].decode('utf-8')  # Decode bytes to string
#         print(f"{category_str:>16s}: {100 * value.item():.2f}%")
        
#     predicted_class = torch.argmax(similarity).item()
#     if label == predicted_class:
#         correct += 1
        
# def data_loader(file_path, batch_size, data_queue):
#     with h5py.File(file_path, 'r') as f:
#         total_size = f['data'].shape[0]
#         for start_idx in range(0, total_size, batch_size):
#             end_idx = min(start_idx + batch_size, total_size)
#             # 从文件中读取数据和标签
#             data_batch = f['data'][start_idx:end_idx]
#             label_batch = f['label'][start_idx:end_idx]
#             # 将数据和标签放入队列
#             data_queue.put((data_batch, label_batch))


# global categories, device, model, preprocess, text_inputs, correct 

# correct = 0

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)
# generator = data_generator(READ_PATH+file, batch_size)

# with h5py.File(READ_PATH+file, 'r') as f:
#     categories = list(f['category'])
# text_inputs = torch.cat([clip.tokenize(f"a volume data of a {c}") for c in categories]).to(device)

# # 创建一个更大的队列和多个加载线程
# data_queue = queue.Queue(maxsize=50)  # 可根据需要调整队列大小
# num_loading_threads = 3  # 加载线程的数量

# loading_threads = [
#     threading.Thread(target=data_loader, args=(READ_PATH+file, batch_size, data_queue))
#     for _ in range(num_loading_threads)
# ]

# # 启动所有加载线程
# for thread in loading_threads:
#     thread.start()

# # 主循环
# for i in range(int(total_samples/batch_size)):
#     voxel_arrays, labels = data_queue.get()
#     for j in range(len(voxel_arrays)):
#         voxel = voxel_arrays[j]
#         label = labels[j]
#         prediction(voxel, label)

# # 等待所有加载线程完成
# for thread in loading_threads:
#     thread.join()

# print(f"Accuracy: {100 * correct / total_samples:.2f}%")
# torch.cuda.empty_cache()
