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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
train_whole_visual_layers = model_parameters['train_whole_visual_layers']

# 创建基于时间戳的目录 for saving model
hyper_param_info = f"_training_{n_way}_way_model_{model_name.replace('/', '_')}_train_whole_visual_layers_lr_{lr}_weight_decay_{weight_decay}_betas_{betas}_eps_{eps}"
timestamp = datetime.now().strftime('%m%d-%H%M%S')
run_dir = f"{SAVE_MODEL_PATH}/"
model_path = os.path.join(run_dir, f"best_model_{timestamp}_{hyper_param_info}.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
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
    if name in train_whole_visual_layers:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)#different layer different learning rate?
# Specify the loss function for images and texts
loss_voxel = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Define scaler for automatic scaling of gradients/another choice is convert model to fp32
scaler = GradScaler()

#data loader h5_file
train_dataset = HDF5_N_WAY_Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='train', num_train_classes=n_way, seed=seed)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

best_valid_loss = float('inf')
# Define scaler for automatic scaling of gradients

writer = SummaryWriter(comment = hyper_param_info)
for epoch in range(num_epochs):
    iterations = len(train_dataloader)
    pbar = tqdm(train_dataloader, total = iterations) #进度条 迭代器，每次迭代立面都有batch_size个元素
    total_loss_accumulated = 0
    #TRAIN
    for batch_index,(voxel_inputs, text_inputs) in enumerate(pbar):# one bach have n elements
        # Zero the gradients
        optimizer.zero_grad()
        voxel_inputs= voxel_inputs.to(device)
    
        # Dynamically tokenize text inputs for each batch
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_inputs]).to(device)
        
        # Enable autocasting for mixed precision
        with autocast():
            #logits_per_voxel的形状：[N, M]，表示N个voxel与M个文本描述之间的相似性分数。
            #logits_per_text的形状：[M, N]，是logits_per_voxel的转置，表示M个文本描述与N个voxel之间的相似性分数。
            #[5, 3, 224, 224, 224]和[5,77]
            logits_per_voxel, logits_per_text = model(voxel_inputs, text_inputs) #会更新模型中所有requires_grad=True的参数，包括logit_scale， 比只用 image_features = model.encode_image(data 和 text 的)更好#Attention, nn.DataParallel(model) may change input dimension due to splitting to different gpu
            
            # Compu te the loss
            # 动态创建ground_truth以匹配每个小批次的维度
            batch_size_actual = logits_per_voxel.shape[0]  # 使用logits_per_voxel的第一维来获取实际的批次大小
            ground_truth = torch.arange(batch_size_actual,dtype=torch.long,device=device)#logits_per_voxel.size(0)动态构建的，这确保了无论在单GPU还是多GPU环境下，ground_truth的大小都与模型输出的批次大小匹配。
            
            # print(f"logits_per_voxel shape: {logits_per_voxel.shape}, dtype: {logits_per_voxel.dtype}")
            # print(f"logits_per_text shape: {logits_per_text.shape}, dtype: {logits_per_text.dtype}")
            # print(f"ground_truth shape: {ground_truth.shape}, dtype: {ground_truth.dtype}")
            
            #how to get local gpu number
            #print(logits_per_voxel.device, logits_per_text.device, ground_truth.device)
            #quit()
            total_loss = (loss_voxel(logits_per_voxel,ground_truth) + loss_txt(logits_per_text,ground_truth))/2# always 2.3, cross entropy.
            
            if torch.isnan(total_loss):
                print(f"NaN detected in loss at epoch {epoch+1}, batch {batch_index+1}")
            # 累积每个批次的损失
            total_loss_accumulated += total_loss.item()
            
        # Perform backward pass and gradient scaling
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # Update model parameters
        scaler.step(optimizer)
        scaler.update()
        
        writer.add_scalar('Loss/train by iterations', total_loss.item(), iterations*epoch + batch_index)
        
        # Print training progress
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")
    
    avg_loss_per_epoch = total_loss_accumulated / iterations
    writer.add_scalar('Loss/train by epoches', avg_loss_per_epoch, epoch)
    # Add your validation logic here
    # Save your model here if it's the best so far
    # VAL.
    # model.eval()  # 设置模型为评估模式
    # total_valid_loss = 0
    # with torch.no_grad():  # 关闭梯度计算
    #     pbar_valid = tqdm(valid_dataloader, total=len(valid_dataloader))
    #     for count, voxel_inputs, text_labels in enumerate(pbar_valid):
    #         voxel_inputs = voxel_inputs.to(device)
    #         text_inputs = torch.cat([clip.tokenize(f"a volume data of a {category}") for category in text_labels]).to(device)

    #         with autocast():
    #             logits_per_voxel, logits_per_text = model(voxel_inputs, text_inputs)
    #             ground_truth = torch.arange(len(voxel_inputs), dtype=torch.long, device=device)
    #             valid_loss = (loss_voxel(logits_per_voxel, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
    #             total_valid_loss += valid_loss.item()
    #             avg_valid_loss = total_valid_loss / (count + 1)
    #         pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_loss.item():.4f}")
    
    
    # # 如果这是迄今为止最佳模型，则保存它
    # if avg_valid_loss < best_valid_loss:
    #     best_valid_loss = avg_valid_loss

    #     torch.save(model.state_dict(), model_path)
    #     print('Model saved as validation loss improved.')
    
    # model.train()  # 设置模型回到训练模式
    
    #save after each epoch training   
    torch.save(model.state_dict(), model_path)
writer.flush()
writer.close()
    