import os
import sys
from clip import clip
import torch 
import h5py
import numpy as np
class_id = 1
file = "voxel.h5"
DATA_PATH = '/data/ScanObjectNN/object_dataset/'
READ_PATH = '/data/ScanObjectNN/processed_object_dataset/'
topk = 5

classes = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# Prepare the prompts
text_inputs = torch.cat([clip.tokenize(f"a volume data of a {c}") for c in classes]).to(device)

#load voxels
with h5py.File(READ_PATH+file, 'r') as f:
  
    # access a specific dataset
    training_data = f['data']# shape: (2309,2048,3) 
  
    # read the data from the dataset
    data = training_data[0]


#for i in range(110, 120):  # Adjust the range based on your needs
# Prepare the inputs
voxel = data#, class_id = stl10_dataset[i]

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
print(f"\n {classes[class_id]} 's top {topk} predictions:")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
torch.cuda.empty_cache()