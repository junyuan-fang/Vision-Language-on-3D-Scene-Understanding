import os
from clip import clip
import torch
from torchvision.datasets import STL10

topk = 5

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
stl10_dataset = STL10(root=os.path.expanduser("./data"), download=False, split='test')
# Prepare the prompts
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in stl10_dataset.classes]).to(device)

for i in range(110, 120):  # Adjust the range based on your needs
    # Prepare the inputs
    image, class_id = stl10_dataset[i]
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Pick the top 5 most similar labels for the image
    values, indices = similarity[0].topk(topk)

    # Print the result
    print(f"\n {stl10_dataset.classes[class_id]} 's top {topk} predictions:")
    for value, index in zip(values, indices):
        print(f"{stl10_dataset.classes[index]:>16s}: {100 * value.item():.2f}%")
torch.cuda.empty_cache()