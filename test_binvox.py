import os

# def get_first_binvox_filenames(root_folder):
#     result = []
#     for category in os.listdir(root_folder):
#         category_path = os.path.join(root_folder, category)
#         if os.path.isdir(category_path):
#             test_folder = os.path.join(category_path, 'test')
#             if os.path.exists(test_folder) and os.path.isdir(test_folder):
#                 binvox_files = [file for file in os.listdir(test_folder) if file.endswith('.binvox')]
#                 if binvox_files:
#                     result.append(os.path.join(test_folder, binvox_files[0]))
#     return result

# # 传入 ModelNet40 文件夹的路径
# modelnet40_folder = 'data/ModelNet40'
# binvox_filenames = get_first_binvox_filenames(modelnet40_folder)

# print(binvox_filenames)