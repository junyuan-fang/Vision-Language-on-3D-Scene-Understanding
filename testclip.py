## need to be moved to CLIP/
# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


import os
from clip import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
#If the original tensor had a shape of [H, W, C], where H is height, W is width, and C is the number of channels (e.g., 3 for RGB),then after unsqueeze(0), the shape becomes [1, H, W, C]. It effectively creates a batch dimension of size 1, 
image_input = preprocess(image).unsqueeze(0).to(device) #224 torch.Size([1, 3, 224, 224])
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)#torch.Size([100, 77])

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)#torch.Size([1, 512])##########
    text_features = model.encode_text(text_inputs)#torch.Size([100, 512]) because here are 100 instances

# 张量中的每个特征向量除以其自身的 L2 范数
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
# Pick the top 5 most similar labels for the image
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)#torch.Size([1, 100])
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

import matplotlib.pyplot as plt
cpu_image = image_input.cpu()
numpy_image = cpu_image[0].permute(1, 2, 0).numpy()

# Plot the NumPy array using imshow
plt.imshow(numpy_image)
plt.axis('off')  # Optional: Turn off axis labels and ticks
plt.show()