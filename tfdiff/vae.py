from sympy import per
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

import torch.nn as nn
import torch.optim as optim
# from dataloader import vaeDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        latent_dim = 26
        hidden_dims = [input_dim, 80, 80]
        # Encoder
        modules = []
        for i in range(len(hidden_dims) - 1):
            in_dim, out_dim = hidden_dims[i], hidden_dims[i + 1]
            modules.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU()))
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)  # 均值
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)  # 方差

        # Decoder
        hidden_dims = [latent_dim, 50, 80, 80]
        modules = []
        in_dim = latent_dim
        for i in range(len(hidden_dims) - 1):
            in_dim, out_dim = hidden_dims[i], hidden_dims[i + 1]
            modules.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.out = nn.Sequential(nn.Linear(hidden_dims[-1], input_dim), nn.Tanh())
    
    def encode(self, x):
        mu = self.fc_mu(self.encoder(x))
        log_var = self.fc_var(self.encoder(x))
        return mu, log_var
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.out(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):

    BCE = nn.functional.mse_loss(recon_x, x)
    #TODO:kld weight
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return BCE + KLD / 1e6

# def train_iter(model, optimizer, train_loader, device):
#     model.train()
#     train_loss = 0
#     with tqdm(total=len(train_loader)) as pbar:
#         for batch_idx, data in enumerate(train_loader):
#             data = data.to(device)
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = model(data)
#             loss = loss_function(recon_batch, data, mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#             pbar.update(1)
#             pbar.set_postfix_str(f'loss:{loss.item():6f}')
#     return train_loss / len(train_loader)

# def train(max_iter, log_dir, data_path, model_path, device):
#     writer = SummaryWriter(log_dir)
#     model = VAE(216).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     dataset = vaeDataset(data_path)
#     print(f'====> Dataset: {len(dataset)}')
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
#     iter = 0
#     while True:
#         # loss = train_iter(model, optimizer, train_loader, device)
#         if max_iter is not None and iter > max_iter:
#             break
#         model.train()
#         # train_loss = 0
#         with tqdm(total=len(train_loader)) as pbar:
#             for batch_idx, data in enumerate(train_loader):
#                 data = data.to(device)
#                 optimizer.zero_grad()
#                 recon_batch, mu, logvar = model(data)
#                 loss = loss_function(recon_batch, data, mu, logvar)
#                 loss.backward()
#                 # train_loss += loss.item()
#                 optimizer.step()
#                 pbar.update(1)
#                 pbar.set_postfix_str(f'loss:{loss.item():6f}')
#                 writer.add_scalar('Loss/train', loss, iter)
#                 iter += 1
#                 if iter % 6000 == 0:
#                     os.makedirs(model_path, exist_ok=True)
#                     torch.save(model.state_dict(), model_path + f'vae_model_{iter}.pth')
#                 writer.flush()
#     writer.close()

# # def sample(model, device, num_samples=64):
# #     model.eval()
# #     with torch.no_grad():
# #         sample = torch.randn(num_samples, 50).to(device)
# #         sample = model.decode(sample).cpu()
# #         result_path = '/home/server1/WQ/SANDiff/sample/'
# #         np.save(result_path + f'sample_vae_exp2_10_8.npy', sample.numpy())

# def sample(model, device, dataset):
#     model.eval()
#     recon = []
#     original = []
#     snr_list = []
#     with torch.no_grad():
#         for data in tqdm(dataset):
#             data = data.to(device)
#             recon_batch, _, _  = model(data)
#             recon.append(recon_batch.cpu().numpy())
#             original.append(data.cpu().numpy())
#             snr = 10 * torch.log10(torch.mean(data ** 2, dim=list(range(len(data.shape)))[1:]) / torch.mean((data - recon_batch) ** 2, dim=list(range(len(data.shape)))[1:]))
#             # print(np.median(snr.cpu().numpy()))
#             snr_list += list(snr.cpu().numpy())
#         recon = np.concatenate(recon, axis=0)
#         original = np.concatenate(original, axis=0)
#         print(np.median(snr_list))
#         result_path = '/media/zuser/Harddisk4TB/WQ/SANDiff/sample/'
#         np.savez(result_path + f'sample_vae_exp_2_15_federated.npz', recon=recon, data=original, snr=np.array(snr_list))

# if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # data_path = [f'/home/server1/WQ/SANDiff/dataset/set1/raw_data/pos{i}' for i in range(1, 11)]
    # data_path = '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1'
    # model_path = '/media/zuser/Harddisk4TB/WQ/SANDiff/model/vae/exp_2_15/'
    # log_dir = '/media/zuser/Harddisk4TB/WQ/SANDiff/vaeLog/exp_2_15/'
    # max_iter = None
    # train(max_iter, log_dir, data_path, model_path, device)
    

    # model = VAE(216).to(device)
    # load_path = '/media/zuser/Harddisk4TB/WQ/SANDiff/model/vae/exp_2_15/vae_model_4566000.pth'
    # # data_path = [f'/home/server1/WQ/SANDiff/dataset/set1/raw_data/pos{i}' for i in range(1, 2)]
    # data_path = '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1'
    # dataset = vaeDataset(data_path, test=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    # model.load_state_dict(torch.load(load_path))
    # model.eval()

    # # Sample from the model
    # sample(model, device, dataloader)

    # # 用上面训练好的vae模型将数据压缩，并将压缩后的数据保存
    # model = VAE(216).to(device)
    # load_path = '/media/zuser/Harddisk4TB/WQ/SANDiff/model/vae/exp_2_15/vae_model_4566000.pth'
    # model.load_state_dict(torch.load(load_path))
    # model.eval()
    # data_path = '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1'
    # dataset = vaeDataset(data_path, test=True)
    # recon, original, snr = sample(model, device, dataset)
    # np.savez(result_path + f'sample_vae_exp_2_15_federated.npz', recon=recon, data=original, snr=np.array(snr))
