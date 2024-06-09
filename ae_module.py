import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F


# from tqdm import tqdm
# from tqdm.notebook import tqdm

from schedulers import ExponentialScheduler

from einops.layers.torch import Reduce
channel_pooling_layer = Reduce('b c h -> b 1 h', 'mean')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def mse_loss(x1, x2):
    return torch.sum((x1 - x2) ** 2)


def weighted_mse_loss(x1, x2, weight):
    return torch.sum(weight * (x1 - x2) ** 2)


def weighted_mse_loss2(x1, x2, weight=1):
    return torch.sum(weight*torch.sum((x1 - x2) ** 2, 1))


def fiber_weight_coeff(alpha, beta, fiber_len=256):
    iter_list = np.array(range(fiber_len))
    weight = (1 / (1+np.exp(-alpha * (iter_list-fiber_len/2)**2 / fiber_len)))**beta
    return weight


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, -1)


class CNN1D_Encoder(nn.Module):
    def __init__(self, in_dim=3, h_dim=4096, z_dim=32, dropoutp=0):

        super(CNN1D_Encoder, self).__init__()

        # 1D CONV for the trajectory
        self.encodernet0 = nn.Sequential(
            nn.Conv1d(in_channels=in_dim,out_channels=32,kernel_size=2,stride=2,padding=0), nn.LeakyReLU(0.2), nn.Dropout(p=dropoutp),
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2,stride=2,padding=0), nn.LeakyReLU(0.2), nn.Dropout(p=dropoutp),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=2,stride=2,padding=0), nn.LeakyReLU(0.2), nn.Dropout(p=dropoutp),
            Flatten(),   
        )

        # bottleneck embedding
        self.fc = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = self.encodernet0(x)
        mu = self.fc(h)

        return mu


class CNN1D_Decoder(nn.Module):
    def __init__(self, out_dim=3, h_dim=4096, z_dim=32, scale=0.001):

        super(CNN1D_Decoder, self).__init__()

        self.scale = scale

        self.fc = nn.Linear(z_dim, h_dim)

        self.decodernet0 = nn.Sequential(
            UnFlatten(),
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=32,out_channels=3,kernel_size=3,stride=1,padding=1), nn.LeakyReLU(0.2),    
            nn.Upsample(scale_factor=2), 
            nn.Conv1d(in_channels=3,out_channels=out_dim,kernel_size=3,stride=1,padding=1), nn.LeakyReLU(0.2),
            # nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        x_bar = self.decodernet0(h)
        # print(x_bar.shape)
        return x_bar 



class CNN1D_AE_Var(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, z_dim=32, dropoutp=0.25, num_pts=256, reduce_mode='reduce', var_mode='log'):
        super(CNN1D_AE_Var, self).__init__()

        h_dim = 16*num_pts
        # h_dim = 4096

        self.decodernet = CNN1D_Decoder(out_dim=out_dim, h_dim=h_dim, z_dim=z_dim)
        self.encodernet = CNN1D_Encoder(in_dim=in_dim, h_dim=h_dim, z_dim=z_dim, dropoutp=dropoutp)
        self.cal_mu = nn.Sequential(
            nn.Linear(num_pts,num_pts),
            # nn.ReLU(),
            # nn.Linear(256,256),
            nn.Sigmoid()
        )
        
        if reduce_mode == 'reduce':
            self.cal_logvar = nn.Sequential(
                nn.Linear(num_pts,num_pts),
                Reduce('b c h -> b 1 h', 'mean')
            )
        else:
            self.cal_logvar = nn.Sequential(
                nn.Linear(num_pts,num_pts)
            )

    def forward(self, x):

        emb = self.encodernet(x)
        x_hat = self.decodernet(emb)
        x_mu = self.cal_mu(x_hat)
        x_logvar = self.cal_logvar(x_hat)
        
        x_var = torch.log(1 + torch.exp(x_logvar)) + 1e-10

        # x_var = x_logvar.exp_() + 1e-10

        return x_mu, x_var 


class CNN1D_AE_Var_Expf(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, z_dim=32, dropoutp=0.25, num_pts=256, reduce_mode='reduce', var_mode='log'):
        super(CNN1D_AE_Var_Expf, self).__init__()

        h_dim = 16*num_pts
        # h_dim = 4096

        self.decodernet = CNN1D_Decoder(out_dim=out_dim, h_dim=h_dim, z_dim=z_dim)
        self.encodernet = CNN1D_Encoder(in_dim=in_dim, h_dim=h_dim, z_dim=z_dim, dropoutp=dropoutp)
        self.cal_mu = nn.Sequential(
            nn.Linear(num_pts,num_pts),
            # nn.ReLU(),
            # nn.Linear(256,256),
            nn.Sigmoid()
        )
        
        if reduce_mode == 'reduce':
            self.cal_logvar = nn.Sequential(
                nn.Linear(num_pts,num_pts),
                Reduce('b c h -> b 1 h', 'mean')
            )
        else:
            self.cal_logvar = nn.Sequential(
                nn.Linear(num_pts,num_pts)
            )

    def forward(self, x):

        emb = self.encodernet(x)
        x_hat = self.decodernet(emb)
        x_mu = self.cal_mu(x_hat)
        x_logvar = self.cal_logvar(x_hat)
        
        # x_var = torch.log(1 + torch.exp(x_logvar)) + 1e-10
        x_var = x_logvar.exp_() + 1e-10

        return x_mu, x_var 


def train(model, dataloader, device, args, writer=None, print_flg=False, verbsome=True):

    if not os.path.exists(args.model_path) or args.force_flg:

        if os.path.exists(args.model_pretrain_path):
            if args.gpuid < 0:
                model.load_state_dict(torch.load(args.model_pretrain_path, map_location=f'cpu'))
            else:
                model.load_state_dict(torch.load(args.model_pretrain_path, map_location=f'cuda:{args.gpuid}'))
                
            if verbsome:
                print(f"Loaded pretrained model from {args.model_pretrain_path}.")       

        # model training

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    #     iteration_idx = 0
        # for epoch in tqdm(range(args.epoch_num)):
        for epoch in range(args.epoch_num):
            total_loss = AverageMeter()
            mse_loss = AverageMeter()

            for batch_idx, (x1, x2, xidx) in enumerate(dataloader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                optimizer.zero_grad()
                
                x1_mu, x1_var = model(x1)
                x2_mu, x2_var = model(x2)
                
                loss1 = torch.mean(0.5*torch.log(x1_var) + 0.5*(torch.square(x1 - x1_mu)/x1_var)) + 4
                loss2 = torch.mean(0.5*torch.log(x2_var) + 0.5*(torch.square(x2 - x2_mu)/x2_var)) + 4

                recon_loss1 = torch.mean(torch.square(x1 - x1_mu))
                recon_loss2 = torch.mean(torch.square(x2 - x2_mu))
                
                loss = loss1 + loss2
                recon_loss = 5000*(recon_loss1 + recon_loss2)

                
                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())
                mse_loss.update(recon_loss.item())
                
            if writer is not None:
                writer.add_scalar(f"ae/total_loss", total_loss.avg, epoch)
                writer.add_scalar(f"ae/mse_loss", mse_loss.avg, epoch)


            if print_flg:
                if epoch % args.update_interval == 0:
                    print(f"epoch {epoch}, total_loss={total_loss.avg:.5f}, mse_loss={mse_loss.avg:.5f}")

                    if epoch > 0:
                        # save trained model
                        inter_model_path = args.model_path0.replace('*', str(epoch))

                        if os.path.exists(inter_model_path):
                            print(f"{inter_model_path} already exits!.")

                        torch.save(model.state_dict(), inter_model_path)
                        print(f"Saved model to {inter_model_path}.")




        # save trained model
        if not os.path.exists(args.model_path):
            torch.save(model.state_dict(), args.model_path)
            print(f"Saved model to {args.model_path}.")
        
        if writer is not None:
            writer.flush()
        
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=f'cuda:{args.gpuid}'))
        if verbsome:
            print(f"Loaded model from {args.model_path}.")       


def pretrain(model, dataloader, device, args, writer=None, print_flg=False, verbsome=True):


    if not os.path.exists(args.model_pretrain_path) or args.force_flg:

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    #     iteration_idx = 0
        # for epoch in tqdm(range(args.pre_epoch_num)):
        for epoch in range(args.pre_epoch_num):
            total_loss = AverageMeter()
            mse_loss = AverageMeter()

            for batch_idx, (x1, x2, xidx) in enumerate(dataloader):
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                optimizer.zero_grad()
                
                x1_mu, x1_var = model(x1)
                x2_mu, x2_var = model(x2)
                
                loss1 = 5000*torch.mean(torch.square(x1 - x1_mu))
                loss2 = 5000*torch.mean(torch.square(x2 - x2_mu))

                recon_loss1 = 5000*torch.mean(torch.square(x1 - x1_mu))
                recon_loss2 = 5000*torch.mean(torch.square(x2 - x2_mu))
                
                loss = loss1 + loss2
                recon_loss = recon_loss1 + recon_loss2

                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())
                mse_loss.update(recon_loss.item())
                
            if writer is not None:
                writer.add_scalar(f"ae/total_loss", total_loss.avg, epoch)
                writer.add_scalar(f"ae/mse_loss", mse_loss.avg, epoch)


            if print_flg:
                if epoch % args.update_interval == 0:
                    print(f"epoch {epoch}, total_loss={total_loss.avg:.5f}, mse_loss={mse_loss.avg:.5f}")

                    if epoch > 0:
                        # save trained model
                        inter_model_pretrain_path = args.model_pretrain_path0.replace('*', str(epoch))

                        if os.path.exists(inter_model_pretrain_path):
                            print(f"{inter_model_pretrain_path} already exits!.")

                        torch.save(model.state_dict(), inter_model_pretrain_path)
                        print(f"Saved model to {inter_model_pretrain_path}.")


        # save trained model
        if not os.path.exists(args.model_pretrain_path):
            torch.save(model.state_dict(), args.model_pretrain_path)
            print(f"Saved model to {args.model_pretrain_path}.")
        
        if writer is not None:
            writer.flush()
        
    else:
        model.load_state_dict(torch.load(args.model_pretrain_path, map_location=device))
        if verbsome:
            print(f"Loaded model from {args.model_pretrain_path}.")        
