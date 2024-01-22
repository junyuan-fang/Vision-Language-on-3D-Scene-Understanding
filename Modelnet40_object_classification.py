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
import binvox_rw
import numpy as np
import argparse
import random
import shutil
def center_print(text):
    terminal_width, _ = shutil.get_terminal_size()
    padding = (terminal_width - len(text)) // 2
    print(" " * padding + text)

def get_all_folder_names(folder_path):
    categories = [category for category in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, category))]
    print("Categories are: ", categories)
    return categories


#Get "ModelNet40" 文件夹下的所有文件夹的 test 文件夹里面获取random ".binx" 文件的名字, 并且 throw to predictions_for_one_voxel() for 运行 prediction
def predictions_for_all_class1x_voxel(root_folder):#root_folder = 'data/ModelNet40'
    random_seed = 42
    random.seed(random_seed)
    result = []
    for category in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):# if it is a folder
            # target is a test folder
            test_folder = os.path.join(category_path, 'test')
            # if test folder exists
            if os.path.exists(test_folder) and os.path.isdir(test_folder): 
                # get all binvox files
                binvox_files = [file for file in os.listdir(test_folder) if file.endswith('.binvox')]
                # if there are binvox files in the test folder
                if binvox_files:
                    random_binvox_file = random.choice(binvox_files)
                    random_binvox_file_path = os.path.join(test_folder, random_binvox_file)
                    print()
                    center_print(f"For foundtruth {category}:")
                    print("Reading file: ", random_binvox_file_path, "...")
                    predictions_for_one_voxel(random_binvox_file_path, top_N)
    return result

def predictions_for_one_voxel(voxel_path, top_N):

    with open(voxel_path, 'rb') as f:#（30,30,30）
        m1 = binvox_rw.read_as_3d_array(f)
        data =m1.data

    preprocessed = preprocess(data)
    voxel_input = preprocessed.unsqueeze(0).to("cuda") #224 torch.Size([1, 3, 224, 224,224])

    given_prompts = get_all_folder_names('data/ModelNet40')#[ "piano", "bathtub","bench","bookshelf", "bottle","xbox"]
    text_inputs = torch.cat([clip.tokenize(f"a image of a {c}") for c in given_prompts]).to(device)#torch.Size([100, 77])

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(voxel_input)#torch.Size([1, 512])##########
        text_features = model.encode_text(text_inputs)#torch.Size([100, 512]) because here are 100 instances

    # 张量中的每个特征向量除以其自身的 L2 范数
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # Pick the top 5 most similar labels for the image
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)#torch.Size([1, 100])
    values, indices = similarity[0].topk(top_N)

    # Print the result

    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        #print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
        print(f"{given_prompts[index]}: {100 * value.item():.2f}%")

def main():######################################################################################################
    parser = argparse.ArgumentParser("Process voxel data classification")
    parser.add_argument("mode", choices=["train", "test"], help="Specify the mode: 'train' or 'test'")
    parser.add_argument("-t","--top_N", type=int, default=40, help="Specify the top N most similar labels to be printed. Default is 40.")
    parser.add_argument("-a", action="store_true", help="Run all tests.")

    args = parser.parse_args()

    # 根据模式执行相应的操作
    if args.mode == "train":
        print("Training mode. ")
        # 在这里添加处理训练模式的逻辑
    elif args.mode == "test":
        print("Testing mode. ")
        # parsing arguments
        global top_N
        top_N = args.top_N


        # Load the model
        global device, model, preprocess
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device)


        if args.a:
            predictions_for_all_class1x_voxel('data/ModelNet40')
        else:
            predictions_for_one_voxel('data/ModelNet40/xbox/test/xbox_0104.binvox', top_N)
    else:
        print("Invalid mode. Use 'train' or 'test'.")

    


if __name__ == "__main__":
    main()
