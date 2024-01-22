import os
from clip import clip
import torch
from torchvision.datasets import STL10
torch.cuda.empty_cache()
topk = 5

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)
print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

# Download the dataset
stl10_dataset = STL10(root=os.path.expanduser("./data"), download=True, split='test')
# Prepare the prompts
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in stl10_dataset.classes]).to(device)
text_inputs = torch.cat([clip.tokenize(f"a volume data of a {c}") for c in stl10_dataset.classes]).to(device)

# Initialize variables for accuracy calculation
correct_predictions = 0
total_samples = 8000

# Iterate over the dataset
for i in range(total_samples):
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

    # # Pick the top 5 most similar labels for the image
    # values, indices = similarity[0].topk(topk)
    # # Print the result
    # print(f"\n {stl10_dataset.classes[class_id]} 's top {topk} predictions:")
    # for value, index in zip(values, indices):
    #     print(f"{stl10_dataset.classes[index]:>16s}: {100 * value.item():.2f}%")
    print(i)

    # Get the predicted class
    predicted_class = torch.argmax(similarity).item()
    # Check if prediction is correct
    if predicted_class == class_id:
        correct_predictions += 1
    

# Compute accuracy
accuracy = correct_predictions / total_samples
print(f"Accuracy on STL-10: {accuracy * 100:.2f}%")
