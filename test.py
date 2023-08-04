import torchsparse
import torch
import torchsparse.nn as spnn
from torch.utils.tensorboard import SummaryWriter

print(torch.cuda.is_available())
print(torchsparse.__version__)


# model = torch.range(0,3*3*3*3-1).reshape(3,3,3,3)
# cout, cin = model.shape[:2]
# squize = model.clone().view(-1,cout, cin) #(3.3.9)


# # squize=weight.clone().view(-1,3,3)#height and width -> one single dimension shape> (3,3,3,3) -> (9,3,3)
# print(squize)