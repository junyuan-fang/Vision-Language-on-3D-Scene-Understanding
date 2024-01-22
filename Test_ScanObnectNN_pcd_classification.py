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
    print(f"\n {categories[label].decode('utf-8')} 's top {topk} predictions:")
    for value, index in zip(values, indices):
        category_str = categories[index].decode('utf-8')  # Decode bytes to string
        print(f"{category_str:>16s}: {100 * value.item():.2f}%")
        
    predicted_class = torch.argmax(similarity).item()
    if label == predicted_class:
        correct += 1

global categories, device, model, preprocess, text_inputs, correct 

correct = 0


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
generator  = data_generator(READ_PATH+file,batch_size)


with h5py.File(READ_PATH+file, 'r') as f:
    categories = list(f['category'])
# Prepare the prompts
print(categories)
print(len(categories))
text_inputs = torch.cat([clip.tokenize(f"a volume data of a {c}") for c in categories]).to(device)



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