#How to fine tune clip： https://www.labellerr.com/blog/fine-tuning-clip-on-custom-dataset/
from clip import clip
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from src.data_loader import HDF5Dataset
from torch.utils.data import DataLoader
import tqdm
#hyperparameters
file = "voxel_all.h5"
num_epochs = 10
log_interval =5
lr=0.001
DATA_READ_PATH = 'data/ScanObjectNN/processed_object_dataset/'
SAVE_MODEL_PATH = 'trained_model'
prompt = "a volume data of a *"
split_ratio = 0.8
seed = 0
batch_size = 10
validation_ratio = 0.1
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6 ,weight_decay=0.2)
betas=(0.9, 0.98)
eps=1e-6
weight_decay=0.2
# Adam optimizer is used with specific hyperparameters
# lr (learning rate) is set to 5e-5, which is considered safe for fine-tuning to a new dataset
# betas are used for the optimization algorithm
# eps is a small value to prevent division by zero
# weight_decay adds L2 regularization to the optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
model = nn.DataParallel(model)  # Wrap the model with DataParallel, to train in parallel 

#fix others' weights but not tunable layers' weights
for name, param in model.named_parameters():
    if name not in ['visual.positional_embedding', 'visual.class_embedding', 'visual.conv1.weight']:
        param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)#different layer different learning rate?
# Specify the loss function for images and texts
loss_voxel = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Define scaler for automatic scaling of gradients/another choice is convert model to fp32
scaler = GradScaler()

#data loader
train_dataset = HDF5Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='train', split_ratio=split_ratio, validation_ratio=validation_ratio,seed=seed)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
valid_dataset = HDF5Dataset(h5_file=DATA_READ_PATH+file, transform=preprocess, tokenization = clip.tokenize, prompt = prompt, split='train', split_ratio=split_ratio, validation_ratio=validation_ratio, seed=seed)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# Define scaler for automatic scaling of gradients
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    #for batch_idx, (data, targets) in enumerate(train_dataloader):
    for voxel_inputs, text_labels in pbar:# one bach have n elements
        # Zero the gradients
        optimizer.zero_grad()
        voxel_inputs= voxel_inputs.to(device)
        
        # Dynamically tokenize text inputs for each batch
        text_inputs = torch.cat([clip.tokenize(f"a volume data of a {category}") for category in text_labels]).to(device)
        
        # Enable autocasting for mixed precision
        with autocast():
            #logits_per_voxel的形状：[N, M]，表示N个voxel与M个文本描述之间的相似性分数。
            #logits_per_text的形状：[M, N]，是logits_per_voxel的转置，表示M个文本描述与N个voxel之间的相似性分数。
            logits_per_voxel, logits_per_text = model(voxel_inputs, text_inputs) #会更新模型中所有requires_grad=True的参数，包括logit_scale， 比只用 image_features = model.encode_image(data 和 text 的)更好
            
            # Compute the loss
            ground_truth = torch.arange(len(voxel_inputs),dtype=torch.long,device=device)
            total_loss = (loss_voxel(logits_per_voxel,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        
        # Perform backward pass and gradient scaling
        scaler.scale(total_loss).backward()
        
        # Update model parameters
        scaler.step(optimizer)
        scaler.update()
        
        # Print training progress
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")
        
        # Add your validation logic here
        # Save your model here if it's the best so far
        # 训练逻辑...
        model.eval()  # 设置模型为评估模式
        total_valid_loss = 0
        with torch.no_grad():  # 关闭梯度计算
            for voxel_inputs, text_labels in valid_dataloader:
                voxel_inputs = voxel_inputs.to(device)
                text_inputs = torch.cat([clip.tokenize(f"a volume data of a {category}") for category in text_labels]).to(device)

                with autocast():
                    logits_per_voxel, logits_per_text = model(voxel_inputs, text_inputs)
                    ground_truth = torch.arange(len(voxel_inputs), dtype=torch.long, device=device)
                    valid_loss = (loss_voxel(logits_per_voxel, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                    total_valid_loss += valid_loss.item()

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        print(f'Validation Loss after Epoch {epoch+1}: {avg_valid_loss:.4f}')

        # 如果这是迄今为止最佳模型，则保存它
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH+"best_model.pth")
            print('Model saved as validation loss improved.')
        
        model.train()  # 设置模型回到训练模式

    
    