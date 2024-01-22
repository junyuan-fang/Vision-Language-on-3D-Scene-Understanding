import os
from clip import clip
import torch
from torchvision.datasets import STL10
torch.cuda.empty_cache()
print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)
print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
# Download the dataset
stl10_dataset = STL10(root=os.path.expanduser("./data"), download=True, split='test')
# Prepare the prompts
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in stl10_dataset.classes]).to(device)

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = len(stl10_dataset)

print(total_samples)
torch.cuda.empty_cache()