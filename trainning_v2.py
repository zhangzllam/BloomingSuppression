#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:58:25 2023

@author: chajaehoon79@gmail.com
"""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm
import numpy as np
import os
import argparse
from collections import OrderedDict

from model_7_7 import CODAE
from calldata import XYRCS,MOOET,MOOET_S
from plots import draw_recover
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='codae')
    parser.add_argument('--path_dir', type=str, default='../')
    parser.add_argument('--dataset', type=str, default='MOOET')
    parser.add_argument('--n_group', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00012)
    parser.add_argument('--input_dim', type=tuple, default=(1, 256, 256))
    parser.add_argument('--hidden_dims', type=tuple, default=(1, 8, 16, 32))
    parser.add_argument('--rhidden_dims', type=tuple, default=(64, 32, 16, 8, 1))
    parser.add_argument('--fc_hidden', type=int, default=128)
    parser.add_argument('--lat_dim', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--w', type=float, default=0.5)
    parser.add_argument('--rnd', type=int, default=0)

    args = parser.parse_args()

    config = OrderedDict([
        ('model_name', args.model_name),
        ('path_dir', args.path_dir),
        ('dataset', args.dataset),
        ('n_group', args.n_group),
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('lr', args.lr),
        ('input_dim', args.input_dim),
        ('hidden_dims', args.hidden_dims),
        ('rhidden_dims', args.rhidden_dims),
        ('fc_hidden', args.fc_hidden),
        ('lat_dim', args.lat_dim),
        ('num_workers', args.num_workers),
        ('warmup', args.warmup),
        ('beta', args.beta),
        ('w', args.w),
        ('rnd', args.rnd),
    ])

    return config



class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = epsilon

    def forward(self, pred, target):
        diff = pred - target
        epsilon = 1e-3
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reproducibility(seed: int):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def total_variation_loss(x):
    loss = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    return loss

def CharbonnierLoss(pred, target, eps = 1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return torch.mean(loss)

def cons_zinv(z_inv,  w):
    #print(z_inv.shape)
    B, C, H, W = z_inv.shape
    z_inv = z_inv.view(B, C, -1)
    loss = 0.0
    for i in range(B):
        for j in range(i + 1, B):
            loss += torch.mean(torch.abs(z_inv[i,0] - z_inv[j,0]))
    loss = loss / (B * (B - 1))
    return loss*w

def cons_Loss(s_hat):
    total_loss = 0
    w = [300]
    #for idx, s_hat in enumerate(S_hat_batch):  # s_hat: [B, C, H, W]
    B, C, H, W = s_hat.shape
    s_hat = s_hat.view(B, C, -1)  # [B, C, H*W]

    # 计算 pairwise 差值矩阵
    s_i = s_hat.unsqueeze(0)      # [1, B, C, HW]
    s_j = s_hat.unsqueeze(1)      # [B, 1, C, HW]
    diff = torch.abs(s_i - s_j)   # [B, B, C, HW]
    diff = diff.mean(dim=-1).mean(dim=-1)  # mean over HW and C → [B, B]
    # 去除对角线（i==j）
    mask = 1 - torch.eye(B, device=s_hat.device)
    loss = (diff * mask).sum() / (B * (B - 1))

    total_loss += w[0] * loss

    return total_loss

def bvae_loss(x, logits, appx_spatial, beta1, beta2):
    recon_x, z, loc, s_hat, b_hat = logits
    #recon_x, z, loc = logits
    #print(z.shape)

    #print(zinv_loss)
    BCE = nn.MSELoss(reduction='none')(recon_x, x).sum((1, 2, 3)).mean()
    #print(BCE) # 8000
    SSIM_loss = 1 - ssim(preds=recon_x, target=x, data_range=1.0)
    #print(SSIM_loss)
    Cons_loss = cons_Loss(s_hat[3])
    #print(Cons_loss) #0.1
    MAE = nn.L1Loss(reduction='none')(loc * weight.to(loc.device), appx_spatial * weight.to(loc.device)).sum(1).mean()
    #print(MAE) #1500
    #0.7 * BCE  + 300 * SSIM_loss + beta2 * MAE + 100 * Cons_loss
    return  (0.7 * BCE  + 300 * SSIM_loss + beta2 * MAE + 100* Cons_loss), (0.6 * BCE  + 300 * SSIM_loss)

def train(epoch, epoch_losses, global_schedule_idx):
    loss_record = []
    epoch_loss = 0.
    epoch_recon_loss = 0.
    model.train()
    for bidx, samples in tqdm(enumerate(train_dataloader)):
        data = Variable(samples['x1']).to(device)
        appx_angles = Variable(samples['aa']).to(device)
        appx_dxs = Variable(samples['ax']).to(device)
        appx_dys = Variable(samples['ay']).to(device)
        appx_spatial = torch.stack((appx_angles, appx_dxs, appx_dys), -1).squeeze(1)

        optimizer.zero_grad()

        outputs = model(data)
        loss, recon_loss = bvae_loss(data.detach().clone(), outputs, appx_spatial, 1.5, beta_exp_scheduler[global_schedule_idx])

        global_schedule_idx += 1

        loss_record.append(loss.detach().cpu().numpy())

        epoch_loss += loss.detach().cpu().numpy()
        epoch_recon_loss += recon_loss.detach().cpu().numpy()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / (bidx + 1)
    epoch_recon_loss = epoch_recon_loss / (bidx + 1)


    print('Train Epoch: {}/{} Loss: {:.4f}   recon_loss:{:.4f}'.format(
        epoch, config["epochs"], epoch_loss, epoch_recon_loss))
    model.eval()
    return loss_record, epoch_loss,  epoch_recon_loss, global_schedule_idx


'''def get_recon_loss():
    loss_record = 0.
    model.eval()
    for bidx, samples in enumerate(train_dataloader):
        data, target = Variable(samples['x1']).to(device), Variable(samples['x1']).to(device)
        output = model(data)
        recons_loss = torch.sum((output - target) ** 2, axis=(1, 2, 3)).mean()
        loss_record += recons_loss.detach().cpu().numpy()
    return loss_record / (bidx + 1)'''


def run(psave_path):
    losses = []
    epoch_losses = [10 ** 15]
    global_schedule_idx = 0

    loss_inital = float('inf')

    for epoch in range(config['epochs']):
        loss_record, epoch_loss,  epoch_recon_loss, schedule_idx = train(epoch, epoch_losses, global_schedule_idx)
        global_schedule_idx = schedule_idx
        losses += loss_record
        epoch_losses.append(epoch_loss)
        if epoch_loss < loss_inital:
            loss_inital = epoch_loss
            torch.save(model.state_dict(), os.path.join(folder_name, psave_path))
            model.load_state_dict(torch.load(os.path.join(folder_name, psave_path)))
            print('model saved')


    return losses


config = parse_args()
reproducibility(config['rnd'])


if config['lat_dim']  == 4:
    W = [[2., 0.5,0.5,0.2]]
elif config['lat_dim']  == 2:
    W = [[2., 0.5,]]
### make folders ###
mother_folder = os.path.join(config['path_dir'], 'results')
try:
    os.mkdir(mother_folder)
except OSError:
    pass

model_name = config['model_name']

for hidx in range(len(config['hidden_dims'])):
    model_name = model_name + '_{}'.format(config['hidden_dims'][hidx])

model_name = model_name + '_{}_{}'.format(config['fc_hidden'], config['lat_dim'])

folder_name = os.path.join(mother_folder, model_name)
try:
    os.mkdir(folder_name)
except OSError:
    pass

for speed in [20]:
    psave_path = rf'model_iter_with_sep_wo_DB_8_15_o3_{speed}hz_all_3.pth'
    #psave_path = rf'full_model_9_21_o3_{speed}hz_1_epoch.pth'
    model = CODAE(alpha=torch.Tensor(W).to(device), config=config, device=device)

    model.to(device)
    '''for name, _ in model.named_parameters():
        print(name)
    print(torch.load(os.path.join(folder_name, 'model_5_30.pth')).keys())'''


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))

    data_path = os.path.join(config['path_dir'], 'data')

    ### call data ###
    train_x = MOOET(data_path, f"MOOET_{speed}hz_o3_all.npz")


    dataset_size = train_x.__len__()

    train_dataloader = DataLoader(train_x, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=config['num_workers'], pin_memory=True, drop_last=True)

    total_iter = config['warmup'] * len(train_dataloader)
    Init_beta = 1.
    decay_rate = 0.1
    min_beta = 0
    weight = torch.Tensor([10., 1000., 1000.])
    beta_exp_scheduler = [(config['beta'] - min_beta) * (1 - decay_rate / len(train_dataloader)) ** i + min_beta for i in
                          range(total_iter)] + [0.] * ((config["epochs"] - config['warmup']) * len(train_dataloader))

    stime = time.time()
    ### implement ###
    losses = run(psave_path)
    etime = time.time()
    training_time = [etime - stime]
    #draw_recover(model, test_dataloader, repre_data, recon_exam, folder_name, device)

    print('Training time: {:.4f}'.format(training_time[0]))



