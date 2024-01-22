import torch
import torchvision.models as models

import torch
from sklearn.decomposition import PCA
import numpy as np
import math


#from torch.Size([768, 3, 32, 32]) to torch.Size([768, 3, 32, 32, 32])
#[outputdim, inputdim, width, height]->[outputdim, inputdim, width, height, depth]
def weight_transform(pre_weight, trans_type='z-axis'): #for conv1
    #from z axis.
    weights = pre_weight.unsqueeze(-1)
    third_dim = pre_weight.shape[-1]
    weights = torch.cat([weights] * third_dim, dim=-1)

    return weights
####################
def weight_transform_3_channels_to_1(pre_weight, trans_type='z-axis'): #for conv1
    # Load your pretrained model or initialize your weight tensor (for demonstration)
    # Assuming your initial weight tensor has the shape: torch.Size([768, 3, 32, 32])

    print( pre_weight.shape)
    # Flatten the kernel weights to apply PCA
    feature_maps = pre_weight.shape[0]
    kernel_size = pre_weight.shape[-1]
    # Shape will be: torch.Size([768, 3 * 32 * 32])
    flattened_weights = pre_weight.view(feature_maps, -1).cpu().numpy()
    # Define the number of components you want (in this case, 1)
    n_components = 1

    # Apply PCA to reduce the number of channels
    pca = PCA(n_components=n_components)
    reduced_weights = pca.fit_transform(flattened_weights)

    # Create the new weight tensor with reduced channels
    # Shape will be: torch.Size([768, 1, 32, 32])
    selected_element = pca.inverse_transform(reduced_weights)
    print(reduced_weights.shape)
    selected_element = torch.from_numpy(selected_element).view(feature_maps, n_components, kernel_size, kernel_size)

    # Move the new weight tensor back to the GPU if necessary
    if torch.cuda.is_available():
        new_weight = new_weight.cuda()
    return new_weight

# from torch.Size([50, 768]). Actually (49+1,768) to (343+1,768)) 
def pos_embed_weight_transform_1(pre_weight):#Inflate by repeating each of the 49 elements seven times and then move to the next element
    tokens , channels = pre_weight.shape
    repeat_factor = int(math.sqrt(tokens-1))
    first_token = pre_weight[0:1]
    repeated_elements = pre_weight[1:].repeat(repeat_factor,1)
    inflated_tensor = torch.cat((first_token,repeated_elements),dim=0)
    return inflated_tensor

def pos_embed_weight_transform_2(pre_weight):#Inflate by repeating each of the 49 elements seven times and then move to the next element
    tokens , channels = pre_weight.shape
    repeat_factor = int(math.sqrt(tokens-1))
    first_token = pre_weight[0:1]
    repeated_elements = pre_weight[1:].repeat_interleave(repeat_factor,dim=0)
    inflated_tensor = torch.cat((first_token,repeated_elements),dim=0)
    return inflated_tensor




# def weight_transform(model, key, trans_type='z-axis'):
#     if 'bn' in key:
#         return model

#     if 'conv' in key and model.shape[-1]==3:
#         cout, cin = model.shape[:2]
#         if trans_type == 'z-axis':
#             model = model.permute(2, 3, 1, 0).contiguous()
#             weights = model.clone().view(-1, cin, cout).repeat(3, 1, 1)
#             weights[:9, :, :] = model[0, :, :, :].repeat(3, 1, 1)
#             weights[9:18, :, :] = model[1, :, :, :].repeat(3, 1, 1)
#             weights[18:, :, :] = model[2, :, :, :].repeat(3, 1, 1)

#         elif trans_type == 'y-axis':
#             trans = torch.eye(9).repeat(1, 3).contiguous()
#             model = model.view(cout, cin, -1)
#             weights = torch.matmul(model, trans).permute(2, 1, 0)

#         elif trans_type == 'x-axis':
#             trans = torch.zeros(9, 27)
#             trans[0, :3] = 1
#             trans[3, 3:6] = 1
#             trans[6, 6:9] = 1
#             trans[1, 9:12] = 1
#             trans[4, 12:15] = 1
#             trans[7, 15:18] = 1
#             trans[2, 18:21] = 1
#             trans[5, 21:24] = 1
#             trans[8, 24:27] = 1
#             trans = trans.contiguous()
#             model = model.view(cout, cin, -1)
#             weights = torch.matmul(model, trans).permute(2, 1, 0)

#     elif model.shape[-1]==1 and 'downsample' in key:
#         weights = model.clone().squeeze().permute(1, 0).contiguous().unsqueeze(0).repeat(8, 1, 1)

#     elif model.shape[-1]==1:
#         weights = model.clone().squeeze().permute(1, 0).contiguous()
#     return weights


# def weight_transform_34(model, key, trans_type='z-axis'):
#     if 'bn' in key:
#         return model
#     if 'conv' in key and model.shape[-1]==3:
#         cout, cin = model.shape[:2]

#         if trans_type == 'z-axis':
#             model = model.permute(2,3,1,0).contiguous()
#             weights = model.clone().view(-1, cin, cout).repeat(3, 1, 1)
#             weights[:9, :, :] = model[0, :, :, :].repeat(3, 1, 1)
#             weights[9:18, :, :] = model[1, :, :, :].repeat(3, 1, 1)
#             weights[18:, :, :] = model[2, :, :, :].repeat(3, 1, 1)

#     elif 'net' in key and 'ds' in key and model.shape[-1]==2:
#         cout, cin = model.shape[:2]
#         model = model.permute(2,3,1,0).contiguous()
#         weights = model.clone().view(-1, cin, cout).repeat(2, 1, 1)
#         weights[:4 :, :] = model[0, :, :, :].repeat(2, 1, 1)
#         weights[4:, :, :] = model[1, :, :, :].repeat(2, 1, 1)

#     elif model.shape[-1]==1 and 'downsample' in key:
#         weights = model.clone().squeeze().permute(1, 0).contiguous()

#     elif model.shape[-1]==1:
#         weights = model.clone().squeeze().permute(1, 0).contiguous()
#     return weights