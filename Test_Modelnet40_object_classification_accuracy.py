
#```
#python3     Test_Modelnet40_object_classification.py  test -a
#```

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
import open3d as o3d
import glob
import shutil
def center_print(text):
    terminal_width, _ = shutil.get_terminal_size()
    padding = (terminal_width - len(text)) // 2
    print(" " * padding + text)

def get_all_folder_names(folder_path):
    categories = [category for category in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, category))]
    print("Categories are: ", categories)
    return categories


# #Get "ModelNet40" 文件夹下的所有文件夹的 test 文件夹里面获取random ".binx" 文件的名字, 并且 throw to predictions_for_one_voxel() for 运行 prediction
# def predictions_for_all_class1x_voxel(root_folder):#root_folder = 'data/ModelNet40'
#     random_seed = 42
#     random.seed(random_seed)
#     result = []
#     for category in os.listdir(root_folder):
#         category_path = os.path.join(root_folder, category)
#         if os.path.isdir(category_path):# if it is a folder
#             # target is a test folder
#             test_folder = os.path.join(category_path, 'test')
#             # if test folder exists
#             if os.path.exists(test_folder) and os.path.isdir(test_folder): 
#                 # get all binvox files
#                 binvox_files = [file for file in os.listdir(test_folder) if file.endswith('.binvox')]
#                 # if there are binvox files in the test folder
#                 if binvox_files:
#                     random_binvox_file = random.choice(binvox_files)
#                     random_binvox_file_path = os.path.join(test_folder, random_binvox_file)
#                     print()
#                     center_print(f"For goundtruth {category}:")
#                     print("Reading file: ", random_binvox_file_path, "...")
#                     predictions_for_one_voxel(random_binvox_file_path, top_N, category)
#     return result
def get_data(root_folder):
    matching_files = []
    labels = []
    label_index = 0
    for category in sorted(os.listdir(root_folder)):
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):# if it is a folder
            # target is a test folder
            test_folder_path = os.path.join(category_path, 'test')
            # if test folder exists
            if os.path.exists(test_folder_path) and os.path.isdir(test_folder_path):
                files = sorted(glob.glob(os.path.join(test_folder_path, '*.binvox')))
                for file_path in files:
                    matching_files = np.append(matching_files,file_path)
                    labels = np.append(labels, label_index)
            label_index += 1
    return matching_files, labels
def process_data (voxel_path):
    with open(voxel_path, 'rb') as f:#（30,30,30） with value True False
        m1 = binvox_rw.read_as_3d_array(f)
        data = m1.data

    return preprocess(data)
def predictions_for_one_voxel(voxel_path, category_index):

    preprocessed = process_data(voxel_path)
    #print(torch.min(preprocessed).item(),torch.max(preprocessed).item())
    voxel_input = preprocessed.unsqueeze(0).to("cuda") #224 torch.Size([1, 3, 224, 224,224])

    text_inputs = torch.cat([clip.tokenize(f"a volume data of a {c}") for c in categories]).to(device)#torch.Size([100, 77])

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(voxel_input)#torch.Size([1, 512])##########
        text_features = model.encode_text(text_inputs)#torch.Size([40, 512]) because here are 100 instances

    # 张量中的每个特征向量除以其自身的 L2 范数
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # Pick the top 5 most similar labels for the image
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)#torch.Size([1, 40])
    values, indices = similarity[0].topk(top_N)#for first picture, top 5 most similar labels

    # Print the result
    center_print(f"For goundtruth {categories[int(category_index)]}:")
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        #print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
        print(f"{categories[index]}: {100 * value.item():.2f}%")

    predicted_class = torch.argmax(similarity).item()
    if category_index == predicted_class:
        return 1
    else:
        return 0
def display_voxel(voxel_array):# for (3,224,224,224)
    new_pcd = o3d.geometry.PointCloud()
    # 遍历体素数组
    points = []
    colors = []
    for x in range(voxel_array.shape[1]):
        for y in range(voxel_array.shape[2]):
            for z in range(voxel_array.shape[3]):
                color = voxel_array[:,x, y, z]
                #if np.any(color == True):  # 检查颜色是否不是全黑
                points.append([x, y, z])
                colors.append(color)
    # 将点和颜色添加到点云对象
    #put color in range 0-1
    color_min = np.min(colors)
    color_max = np.max(colors)
    colors = (colors-color_min)/(color_max-color_min)
    print(np.array(points).shape)
    print(np.array(colors).shape)

    new_pcd.points = o3d.utility.Vector3dVector(np.array(points))  # 缩放点到原始大小
    new_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    print(set(color))
    o3d.visualization.draw_geometries([new_pcd])
def main():######################################################################################################
    length = 100
    print("Testing mode. ")
    # parsing arguments
    global top_N
    top_N = 5
    DATA_PATH = 'data/ModelNet40'
    # Load the model
    global device, model, preprocess, categories
    correct =0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    categories  = get_all_folder_names(DATA_PATH)#different categories such as [ "piano", "bathtub","bench","bookshelf", "bottle", "xbox", ...]
    matching_files, labels = get_data(DATA_PATH)
    

    np.random.seed(123)
    indices = np.arange(len(matching_files))
    np.random.shuffle(indices)
    matching_files = np.array(matching_files)[indices]
    labels = np.array(labels)[indices]

    matching_files =  matching_files[:length]
    labels =  labels[:length]

    for voxel_path, category_index in zip(matching_files, labels):
        print("Reading file: ", voxel_path, "...")
        correct+=predictions_for_one_voxel(voxel_path, category_index)
        #display_voxel(voxel_array=process_data (voxel_path))
    print("Accuracy is: ", correct/len(labels))


    


if __name__ == "__main__":
    main()
