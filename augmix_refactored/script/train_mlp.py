import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from augmix_refactored.models.mlp import MLP

from datetime import datetime
import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learning-rate", default=0.5, type=float)
args = parser.parse_args()
random.seed(0)
cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

initial_lr= args.learning_rate
save_path = "/home/shashank/Projects/augmix-refactored/snapshots/linear_act/" + "lr_"+str(initial_lr) + '_'+ datetime.now().strftime("%y-%m-%d_%H_%M_%S_%f") + '/'

os.makedirs(save_path, exist_ok=True)

writer = SummaryWriter(log_dir = save_path)

layer64 = nn.Linear(64,64,bias=True).to(device)
layer128 = nn.Linear(128,128,bias=True).to(device)
layer256 = nn.Linear(256,256,bias=True).to(device)
layer512 = nn.Linear(512,512,bias=True).to(device)
mlp = MLP(width=3, depth=3, in_features=64, out_features=64).to(device)
mlp_256_3 = MLP(width=256, depth=3, in_features=64, out_features=64).to(device)
mlp_256_5 = MLP(width=256, depth=5, in_features=64, out_features=64).to(device)
mlp_512_5 = MLP(width=512, depth=5, in_features=64, out_features=64).to(device)
mlp_512_10 = MLP(width=512, depth=10, in_features=64, out_features=64).to(device)
"""
layer64 = nn.LazyLinear(64,bias=True).to(device)
layer128 = nn.LazyLinear(128,bias=True).to(device)
layer256 = nn.LazyLinear(256,bias=True).to(device)
layer512 = nn.LazyLinear(512,bias=True).to(device)
"""
criterion = nn.CrossEntropyLoss()

optim_mlp = torch.optim.SGD(mlp.parameters(), lr=initial_lr, weight_decay=0.0001)
optim_mlp_256_3 = torch.optim.SGD(mlp_256_3.parameters(), lr=initial_lr, weight_decay=0.0001)
optim_mlp_256_5 = torch.optim.SGD(mlp_256_5.parameters(), lr=initial_lr, weight_decay=0.0001)
optim_mlp_512_5 = torch.optim.SGD(mlp_512_5.parameters(), lr=initial_lr, weight_decay=0.0001)
optim_mlp_512_10 = torch.optim.SGD(mlp_512_10.parameters(), lr=initial_lr, weight_decay=0.0001)
optim64 = torch.optim.SGD(layer64.parameters(), lr=initial_lr, weight_decay=0.0001) #, momentum= 0.9)
optim128 = torch.optim.SGD(layer128.parameters(), lr=initial_lr, weight_decay=0.0001) #, momentum= 0.9)
optim256 = torch.optim.SGD(layer256.parameters(), lr=initial_lr, weight_decay=0.0001) #, momentum= 0.9)
optim512 = torch.optim.SGD(layer512.parameters(), lr=initial_lr, weight_decay=0.0001) #, momentum= 0.9)

scheduler_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mlp, T_max=1000000)
scheduler_mlp_256_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mlp_256_3, T_max=1000000)
scheduler_mlp_256_5 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mlp_256_5, T_max=1000000)
scheduler_mlp_512_5 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mlp_512_5, T_max=1000000)
scheduler_mlp_512_10 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mlp_512_10, T_max=1000000)
scheduler64 = torch.optim.lr_scheduler.CosineAnnealingLR(optim64, T_max=1000000)
scheduler128 = torch.optim.lr_scheduler.CosineAnnealingLR(optim128, T_max=1000000)
scheduler256 = torch.optim.lr_scheduler.CosineAnnealingLR(optim256, T_max=1000000)
scheduler512 = torch.optim.lr_scheduler.CosineAnnealingLR(optim512, T_max=1000000)

gelu = nn.GELU()
pbar = tqdm(range(1000000))

for i in pbar:
    optim_mlp.zero_grad()
    optim_mlp_256_3.zero_grad()
    optim_mlp_256_5.zero_grad()
    optim_mlp_512_5.zero_grad()
    optim_mlp_512_10.zero_grad()
    optim64.zero_grad()
    optim128.zero_grad()
    optim256.zero_grad()
    optim512.zero_grad()

    random_pos = torch.rand(256,32,32).uniform_(-10, 100)
    random_neg = torch.rand(256,32,32).uniform_(-100, 10)
    random_num = torch.concat([random_pos, random_neg])
    idx = torch.randperm(random_num.shape[0])
    random_num = random_num[idx].view(random_num.size()).to(device)

    #import ipdb;ipdb.set_trace()
    output_mlp = mlp(random_num[:64,:,:].view(1024,64)).view(64,32,32)
    output_mlp_256_3 = mlp_256_3(random_num[:64,:,:].view(1024,64)).view(64,32,32)
    output_mlp_256_5 = mlp_256_5(random_num[:64,:,:].view(1024,64)).view(64,32,32)
    output_mlp_512_5 = mlp_512_5(random_num[:64,:,:].view(1024,64)).view(64,32,32)
    output_mlp_512_10 = mlp_512_10(random_num[:64,:,:].view(1024,64)).view(64,32,32)
    output64 = layer64(random_num[:64,:,:].view(1024,64)).view(64,32,32)
    output128 = layer128(random_num[:128,:,:].view(1024,128)).view(128,32,32)
    output256 = layer256(random_num[:256,:,:].view(1024,256)).view(256,32,32)
    output512 = layer512(random_num.view(1024,512)).view(512,32,32)

    loss_mlp = criterion(output_mlp, gelu(random_num[:64,:,:]))
    loss_mlp_256_3 = criterion(output_mlp_256_3, gelu(random_num[:64,:,:]))
    loss_mlp_256_5 = criterion(output_mlp_256_5, gelu(random_num[:64,:,:]))
    loss_mlp_512_5 = criterion(output_mlp_512_5, gelu(random_num[:64,:,:]))
    loss_mlp_512_10 = criterion(output_mlp_512_10, gelu(random_num[:64,:,:]))
    #if torch.isnan(loss_mlp):
    #   import ipdb;ipdb.set_trace()
    loss64 = criterion(output64, gelu(random_num[:64,:,:]))
    loss128 = criterion(output128, gelu(random_num[:128,:,:]))
    loss256 = criterion(output256, gelu(random_num[:256,:,:]))
    loss512 = criterion(output512, gelu(random_num))

    acc_mlp = (torch.sum(output_mlp==random_num[:64,:,:]).item())/output_mlp.numel()
    acc_mlp_256_3 = (torch.sum(output_mlp_256_3==random_num[:64,:,:]).item())/output_mlp.numel()
    acc_mlp_256_5 = (torch.sum(output_mlp_256_5==random_num[:64,:,:]).item())/output_mlp.numel()
    acc_mlp_512_5 = (torch.sum(output_mlp_512_5==random_num[:64,:,:]).item())/output_mlp.numel()
    acc_mlp_512_10 = (torch.sum(output_mlp_512_10==random_num[:64,:,:]).item())/output_mlp.numel()
    acc64 = (torch.sum(output64==random_num[:64,:,:]).item())/output64.numel()
    acc128 = (torch.sum(output128==random_num[:128,:,:]).item())/output128.numel()
    acc256 = (torch.sum(output256==random_num[:256,:,:]).item())/output256.numel()
    acc512 = (torch.sum(output512==random_num).item())/output512.numel()

    mean_l2_mlp = torch.cdist(output_mlp, random_num[:64,:,:], p=2).mean()
    mean_l2_mlp_256_3 = torch.cdist(output_mlp_256_3, random_num[:64,:,:], p=2).mean()
    mean_l2_mlp_256_5 = torch.cdist(output_mlp_256_5, random_num[:64,:,:], p=2).mean()
    mean_l2_mlp_512_5 = torch.cdist(output_mlp_512_5, random_num[:64,:,:], p=2).mean()
    mean_l2_mlp_512_10 = torch.cdist(output_mlp_512_10, random_num[:64,:,:], p=2).mean()
    mean_l2_64 = torch.cdist(output64, random_num[:64,:,:], p=2).mean()
    mean_l2_128 = torch.cdist(output128, random_num[:128,:,:], p=2).mean()
    mean_l2_256 = torch.cdist(output256, random_num[:256,:,:], p=2).mean()
    mean_l2_512 = torch.cdist(output512, random_num, p=2).mean()

    pbar.set_postfix_str("LR:{} || MLP Loss: {} Acc: {}.|.Layer 64 Loss: {} Acc: {}.|.Layer 128 Loss: {} Acc: {}.|.Layer 256 Loss: {} Acc: {}.|.Layer 512 Loss: {} Acc: {}.|.".format(scheduler_mlp.get_last_lr()[-1],loss_mlp, acc_mlp, loss64, acc64,loss128,acc128,loss256,acc256,loss512,acc512))
    writer.add_scalar('MLP Width 512 Depth 5/loss', loss_mlp_512_5, i)
    writer.add_scalar('MLP Width 512 Depth 5/accuracy', acc_mlp_512_5, i)
    writer.add_scalar('MLP Width 512 Depth 5/mean l2', mean_l2_mlp_512_5, i)

    writer.add_scalar('MLP Width 512 Depth 10/loss', loss_mlp_512_10, i)
    writer.add_scalar('MLP Width 512 Depth 10/accuracy', acc_mlp_512_10, i)
    writer.add_scalar('MLP Width 512 Depth 10/mean l2', mean_l2_mlp_512_10, i)

    writer.add_scalar('MLP Width 256 Depth 3/loss', loss_mlp_256_3, i)
    writer.add_scalar('MLP Width 256 Depth 3/accuracy', acc_mlp_256_3, i)
    writer.add_scalar('MLP Width 256 Depth 3/mean l2', mean_l2_mlp_256_3, i)

    writer.add_scalar('MLP Width 256 Depth 5/loss', loss_mlp_256_5, i)
    writer.add_scalar('MLP Width 256 Depth 5/accuracy', acc_mlp_256_5, i)
    writer.add_scalar('MLP Width 256 Depth 5/mean l2', mean_l2_mlp_256_5, i)

    writer.add_scalar('MLP Width 3 Depth 3/loss', loss_mlp, i)
    writer.add_scalar('MLP Width 3 Depth 3/accuracy', acc_mlp, i)
    writer.add_scalar('MLP Width 3 Depth 3/mean l2', mean_l2_mlp, i)

    writer.add_scalar('Linear64/loss', loss64, i)
    writer.add_scalar('Linear64/accuracy', acc64, i)
    writer.add_scalar('Linear64/mean l2', mean_l2_64, i)

    writer.add_scalar('Linear128/loss', loss128, i)
    writer.add_scalar('Linear128/accuracy', acc128, i)
    writer.add_scalar('Linear128/mean l2', mean_l2_128, i)

    writer.add_scalar('Linear256/loss', loss256, i)
    writer.add_scalar('Linear256/accuracy', acc256, i)
    writer.add_scalar('Linear256/mean l2', mean_l2_256, i)

    writer.add_scalar('Linear512/loss', loss512, i)
    writer.add_scalar('Linear512/accuracy', acc512, i)
    writer.add_scalar('Linear512/mean l2', mean_l2_512, i)
    #import ipdb;ipdb.set_trace()
    writer.add_scalar('Learning Rate', scheduler_mlp.get_last_lr()[-1], i)

    loss_mlp.backward()
    loss_mlp_256_3.backward()
    loss_mlp_256_5.backward()
    loss_mlp_512_5.backward()
    loss_mlp_512_10.backward()
    loss64.backward()
    loss128.backward()
    loss256.backward()
    loss512.backward()

    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(mlp_256_3.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(mlp_256_5.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(mlp_512_5.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(mlp_512_10.parameters(), 0.5)

    optim_mlp.step()
    optim_mlp_256_3.step()
    optim_mlp_256_5.step()
    optim_mlp_512_5.step()
    optim_mlp_512_10.step()
    optim64.step()
    optim128.step()
    optim256.step()
    optim512.step()

    scheduler_mlp.step()
    scheduler_mlp_256_3.step()
    scheduler_mlp_256_5.step()
    scheduler_mlp_512_5.step()
    scheduler_mlp_512_10.step()
    scheduler64.step()
    scheduler128.step()
    scheduler256.step()
    scheduler512.step()

torch.save(mlp, save_path+'mlp_3_3.pt')
torch.save(mlp_256_3, save_path+'mlp_256_3.pt')
torch.save(mlp_256_5, save_path+'mlp_256_5.pt')
torch.save(mlp_512_5, save_path+'mlp_512_5.pt')
torch.save(mlp_512_10, save_path+'mlp_512_10.pt')
torch.save(layer64, save_path+'layer64.pt')
torch.save(layer128, save_path+'layer128.pt')
torch.save(layer256, save_path+'layer256.pt')
torch.save(layer512, save_path+'layer512.pt')



