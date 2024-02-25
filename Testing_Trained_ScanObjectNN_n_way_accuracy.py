#How to fine tune clip： https://www.labellerr.com/blog/fine-tuning-clip-on-custom-dataset/
from clip import clip
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from src.data_loader import HDF5_N_WAY_Dataset
from torch.utils.data import DataLoader
from  tqdm import tqdm
import yaml
import os
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train or test the model based on HDF5 dataset.")
parser.add_argument('--mode', type=str, choices=['test_train', 'test_test', 'test_valid'], required=True,
                    help='Mode to run the script in. "train" for training mode, "test" for testing mode.')
args = parser.parse_args()

# Load configurations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debugging
#hyperparameters
with open('config/train_n_way.yml', 'r') as file:
    config = yaml.safe_load(file)

# 加载数据参数
data_parameters = config['data_parameters']
file = data_parameters['file']
DATA_READ_PATH = data_parameters['DATA_READ_PATH']
prompt = data_parameters['prompt']
n_way = data_parameters['n_way']
seed = data_parameters['seed']
batch_size = data_parameters['batch_size']
validation_ratio = data_parameters['validation_ratio']

# 加载模型参数
model_parameters = config['model_parameters']
model_name = model_parameters['model_name']
num_epochs = int(model_parameters['num_epochs'])
log_interval = int(model_parameters['log_interval'])
lr = float(model_parameters['lr'])
SAVE_MODEL_PATH = model_parameters['SAVE_MODEL_PATH']
optimizer_config = model_parameters['optimizer']
betas = tuple([float(beta) for beta in optimizer_config['betas']]) 
eps = float(optimizer_config['eps'])
weight_decay = float(optimizer_config['weight_decay'])
train_layers = model_parameters['train_layers']

# 创建基于时间戳的目录 for saving model
hyper_param_info = f"_model_{model_name.replace('/', '_')}_lr_{lr}_weight_decay_{weight_decay}_betas_{betas}_eps_{eps}"
timestamp = datetime.now().strftime('%m%d-%H%M%S')
run_dir = f"{SAVE_MODEL_PATH}/"
model_path = os.path.join(run_dir, f"best_model_{timestamp}_{hyper_param_info}.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Ensure model's data types are consistent for mixed precision training
#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
convert_models_to_fp32(model)
#model = nn.DataParallel(model).to(device)  # Wrap the model with DataParallel, to train in parallel 

#fix others' weights but not tunable layers' weights
for name, param in model.named_parameters():
    if name in train_layers:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)#different layer different learning rate?
# Specify the loss function for images and texts
loss_voxel = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
scaler = GradScaler()# Define scaler for automatic scaling of gradients/another choice is convert model to fp32

if args.mode == 'test_train':
    #train_dataset = HDF5Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='train', split_ratio=split_ratio, validation_ratio=validation_ratio,seed=seed)
    train_dataset = HDF5_N_WAY_Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='train', num_train_classes=n_way, seed=seed)

    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True) 

elif args.mode == 'test_test':
    test_dataset = HDF5_N_WAY_Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='test', num_train_classes=n_way, seed=seed)
    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
#elif args.mode == 'test_valid':
    #valid_dataset = HDF5Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='valid', split_ratio=split_ratio, validation_ratio=validation_ratio, seed=seed)
    ##dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=False)
    
# Load the model
model_path = 'trained_model/best_model_0223-205657__training_ratio0.2_model_ViT-B_32_train_whole_visual_layers_lr_1e-05_weight_decay_0.2_betas_(0.9, 0.98)_eps_1e-06.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
    # Test loop
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    pbar = tqdm(dataloader, total = len(dataloader))
    for voxel_inputs, text_inputs in dataloader:
        voxel_inputs = voxel_inputs.to(device)
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_inputs]).to(device)
        logits_per_voxel, logits_per_text = model(voxel_inputs, text_inputs)
        _, predicted = torch.max(logits_per_voxel, 1)# Returns the maximum value of each row of the input tensor in the given dimension dim, the second return value is the index location of each maximum value found (argmax).
        total += voxel_inputs.size(0)
        correct += (predicted == torch.arange(len(voxel_inputs), device=device)).sum().item()
        pbar.update(1)  # Update the progress bar
        pbar.set_postfix({'Accuracy': f"{100 * correct / total:.2f}%"})


    