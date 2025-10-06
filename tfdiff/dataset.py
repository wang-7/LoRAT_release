import einops
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from tfdiff.params import AttrDict
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from tfdiff.vae import VAE
# data_key='csi_data',
# gesture_key='gesture',
# location_key='location',
# orient_key='orient',
# room_key='room',
# rx_key='rx',
# user_key='user',


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)

def name2cond(filename):
    cond = [1]
    for i in range(3):
        cond.append(int(filename[-14+2*i]))
    cond.append(int(filename[-5]))
    cond.append(int(filename[-16]))
    return torch.tensor([cond]).to(torch.complex64)

class WiFiDataset(torch.utils.data.Dataset):
    def __init__(self, paths, data_key, cond_key):
        super().__init__()
        self.filenames = []
        self.data_key = data_key
        self.cond_key = cond_key
        for path in paths:
            self.filenames += glob(f'{path}/**/user*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        # cur_data = torch.from_numpy(cur_sample['csi_data']).to(torch.complex64)
        cur_data = torch.from_numpy(cur_sample[self.data_key]).to(torch.complex64)
        if self.cond_key == 'None':
            cur_cond = name2cond(cur_filename)
        else:
            cur_cond = torch.from_numpy(cur_sample[self.cond_key]).to(torch.complex64)
        return {
            'data': cur_data,
            'cond': cur_cond.squeeze(0)
        }
    

class FMCWDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename)
        cur_data = torch.from_numpy(cur_sample['feature']).to(torch.complex64)
        cur_cond = torch.from_numpy(cur_sample['cond'].astype(np.int16)).to(torch.complex64)
        return {
            'data': cur_data,
            'cond': cur_cond.squeeze(0)
        }

class MIMODataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
        self.filenames += glob(f'{path}/**/*.mat', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self,idx):
    dataset = scio.loadmat(self.filenames[idx])
    data = torch.from_numpy(dataset['down_link']).to(torch.complex64)
    cond = torch.from_numpy(dataset['up_link']).to(torch.complex64)
    return {
        'data': torch.view_as_real(data),
        'cond': torch.view_as_real(cond)
    }


class EEGDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    paths = paths[0]
    self.filenames = []
    self.filenames += glob(f'{paths}/*.mat', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self,idx):
    dataset = scio.loadmat(self.filenames[idx])
    data = torch.from_numpy(dataset['clean']).to(torch.complex64)
    cond = torch.from_numpy(dataset['disturb']).to(torch.complex64)
    return {
        'data': data,
        'cond': cond
    }

class coordCSIDataset(torch.utils.data.Dataset):
    def __init__(self, load_path):
        super().__init__()
        self.filenames = []
        self.filenames += glob(f'{load_path}/**/*.mat', recursive=True)
        # self.load_path = load_path
        
    def __len__(self):
        # loaded = np.load(self.load_path, allow_pickle=True)
        return len(self.filenames)
    def __getitem__(self, idx):
        # loaded = np.load(self.filenames[idx], allow_pickle=True)
        loaded = scio.loadmat(self.filenames[idx])
        rxpos, est= \
        loaded['pos_i'], loaded['chanEst'][::4, :200]
        data = torch.from_numpy(est).to(torch.complex64)
        cond = torch.from_numpy(rxpos).to(torch.complex64)
        return {
            'data': torch.view_as_real(data),
            'cond': torch.view_as_real(cond)
        }


class diffusionDataset(torch.utils.data.Dataset):
    def __init__(self, load_path):
        super().__init__()
        self.data = np.load(load_path, allow_pickle=True)
        est = self.data['est'][:, :, :200]
        pos = self.data['pos'].reshape(-1, 3)
        velocity = self.data['velocity'].reshape(-1, 3)
        # TODO: if the shape of the data reasonable?
        est = rearrange(est, 'num subcarrier time tx rx -> num time (tx rx subcarrier)')
        # est = np.concatenate((est.real, est.imag), axis=-1)
        est = est / abs(est).max()
        # normalize pos and velocity
        pos = (pos - pos.mean(axis=0)) / pos.std(axis=0)
        velocity = (velocity - velocity.mean(axis=0)) / velocity.std(axis=0)
        self.data = torch.from_numpy(est).to(torch.complex64)
        self.cond = torch.from_numpy(np.concatenate((pos, velocity), axis=1)).to(torch.complex64)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return {
            'data': torch.view_as_real(self.data[idx]),
            'cond': torch.view_as_real(self.cond[idx])
        }

class exp_2_12Dataset(torch.utils.data.Dataset):
    def __init__(self, load_path):
        super().__init__()
        self.filenames = []
        for path in load_path:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        # cur_data = torch.from_numpy(cur_sample['csi_data']).to(torch.complex64)
        up_link = cur_sample['chanEst'][:108]
        down_link = cur_sample['chanEst'][108:]
        cur_data = torch.from_numpy(down_link).to(torch.complex64)
        cur_cond = torch.from_numpy(up_link).to(torch.complex64)
        # print(cur_data.shape)
        return {
            'data': cur_data,
            'cond': cur_cond
        }
    
class exp_2_16Dataset(torch.utils.data.Dataset):
    # vae + federated
    def __init__(self, load_path, vae_path):
        super().__init__()
        self.filenames = []
        for path in load_path:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)
        self.vae_model = VAE(216)
        self.vae_model.load_state_dict(torch.load(vae_path))
        self.vae_model.eval()
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        # cur_data = torch.from_numpy(cur_sample['csi_data']).to(torch.complex64)
        # Process uplink data
        up_link_complex = cur_sample['chanEst'][:108]
        up_link = np.concatenate([np.real(up_link_complex), np.imag(up_link_complex)], axis=0)
        up_link = torch.from_numpy(up_link.T).to(torch.float32)
        
        # Process downlink data 
        down_link_complex = cur_sample['chanEst'][108:]
        down_link = np.concatenate([np.real(down_link_complex), np.imag(down_link_complex)], axis=0)
        down_link = torch.from_numpy(down_link.T).to(torch.float32)

        # Normalize and encode through VAE
        norm_max_constant = 1.5242553267799146
        with torch.no_grad():
            # Encode uplink
            up_latent = self.vae_model.reparameterize(
                *self.vae_model.encode(up_link / norm_max_constant)
            )
            # Encode downlink
            down_latent = self.vae_model.reparameterize(
                *self.vae_model.encode(down_link / norm_max_constant)
            )

        # Convert to complex tensors
        cur_data = down_latent.T.to(torch.complex64)
        cur_cond = up_latent.T.to(torch.complex64)
        return {
            'data': cur_data,
            # NOTE：原代码'cond': cur_cond，改为不经过vae编码的cond
            'cond': torch.from_numpy(up_link_complex).to(torch.complex64)
        }
    
class exp_3_7_Argos(torch.utils.data.Dataset):
  def __init__(self, paths, lora=False, FT_id=None, seen_idx=None, ant_idx=None):
    super().__init__()
    # original shape:time*user*ant*subcarrier
    # seen_idx = np.array([1, 2, 3, 4, 5, 6, 7])
    seen_idx = np.array(seen_idx)
    # if lora:
    #     self.data = einops.rearrange(np.load(paths[0])[:, FT_id, :, :], '(num time) ant subcarrier -> num time ant subcarrier', time=14)
    # else:
    #     self.data = einops.rearrange(np.load(paths[0])[:, seen_idx, :, :], '(num time) user ant subcarrier -> (user num) time ant subcarrier', time=14)
    if lora:
        self.data = einops.rearrange(np.load(paths[0])[:, FT_id, 2:3, :], '(num time) ant subcarrier -> num time ant subcarrier', time=14)
    else:
        self.data = einops.rearrange(np.load(paths[0])[:, seen_idx, 2:3, :], '(num time) user ant subcarrier -> (user num) time ant subcarrier', time=14)

    # self.filenames = []
    # for path in paths:
    #     self.filenames += glob(f'{path}/**/*.mat', recursive=True)

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    # dataset = scio.loadmat(self.filenames[idx])
    data = torch.from_numpy(self.data[idx, ..., 26:]).to(torch.complex64)
    cond = torch.from_numpy(self.data[idx, ..., :26]).to(torch.complex64)
    return {
        'data': torch.view_as_real(data),
        'cond': torch.view_as_real(cond)
    }

class exp_vivo(torch.utils.data.Dataset):
  def __init__(self, paths, lora=False, FT_id=None, seen_idx=None, ant_idx=None):
    super().__init__()
    # original shape:time*user*ant*subcarrier
    # seen_idx = np.array([1, 2, 3, 4, 5, 6, 7])
    seen_idx = np.array(seen_idx)

    if lora:
        self.data = einops.rearrange(np.load(paths[0])[:, FT_id, 0:1, :], '(num time) ue (ant freq) -> (ue num) time ant freq', time=20, freq=1)
    else:
        self.data = einops.rearrange(np.load(paths[0])[:, seen_idx, 0:1, :], '(num time) sector ue (ant freq) -> (sector ue num) time ant freq', time=20, freq=1)

    # if lora:
    #     self.data = einops.rearrange(np.load(paths[0])[:, FT_id, :, :], '(num time) ue ant -> num time ue ant', time=20)
    # else:
    #     self.data = einops.rearrange(np.load(paths[0])[:, seen_idx, :, :], '(num time) sector ue ant -> (sector num) time ue ant', time=20)

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    # dataset = scio.loadmat(self.filenames[idx])
    data = torch.from_numpy(self.data[idx, 10:, :, :]).to(torch.complex64)
    cond = torch.from_numpy(self.data[idx, :10, :, :]).to(torch.complex64)
    return {
        'data': torch.view_as_real(data),
        'cond': torch.view_as_real(cond)
    }

class exp_4_3_multiband(torch.utils.data.Dataset):
  def __init__(self, data_paths):
    super().__init__()
    self.data_filename = []
    self.cond_filename = []
    for path in data_paths:
        self.data_filename += glob(f'{path}/data/**/*.npy', recursive=True)
    for path in data_paths:
        self.cond_filename += glob(f'{path}/cond/**/*.mat', recursive=True)
    self.data_filename.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    self.cond_filename.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    # data.shape = [900, 5, 216]
  def __len__(self):
    return len(self.data_filename)

  def __getitem__(self,idx):
    #未转置： [216, 1]
    cond = scio.loadmat(self.cond_filename[idx])['chanEst']
    # [5, 216]
    data = np.load(self.data_filename[idx])
    if len(data.shape) == 2:
        data = torch.from_numpy(data).unsqueeze(0).to(torch.complex64)
    elif len(data.shape) == 1:
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(torch.complex64)
    else:
        raise ValueError("Unexpected data shape.")
    cond = torch.from_numpy(cond).unsqueeze(0).to(torch.complex64)
    return {
        'data': torch.view_as_real(data),
        'cond': torch.view_as_real(cond)
    }
  

class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        sample_rate = self.params.sample_rate
        task_id = self.params.task_id
        ## WiFi Case
        if task_id == 0:
            for record in minibatch:
                # Filter out records that aren't long enough.
                if len(record['data']) < sample_rate:
                    del record['data']
                    del record['cond']
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }
        ## FMCW Case
        elif task_id == 1:
            for record in minibatch:
                # Filter out records that aren't long enough.
                if len(record['data']) < sample_rate:
                    del record['data']
                    del record['cond']
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }

        ## MIMO Case
        elif task_id in [2, 10, 11, 12]:
            for record in minibatch:
                data = record['data']
                cond = record['cond']
                # print(f'data.shape:{data.shape}')
                norm_data = (data) / data.std()
                norm_cond = (cond) / cond.std()
                # print(f'norm_data.shape:{norm_data.shape}')
                record['data'] = norm_data.transpose(1,2)
                record['cond'] = norm_cond.transpose(1,2)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': cond,
            } 

        ## EEG Case
        elif task_id == 3:
            for record in minibatch:
                data = record['data']
                cond = record['cond']

                norm_data = data / cond.std()
                norm_cond = cond / cond.std()
                
                record['data'] = norm_data.reshape(512, 1, 1)
                record['cond'] = norm_cond.reshape(512)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': torch.view_as_real(data),
                'cond': torch.view_as_real(cond),
            } 

        elif task_id == 4:
            for record in minibatch:
                data = record['data']
                cond = record['cond']
                # print(f'data.shape:{data.shape}')
                norm_data = (data - data.mean()) / data.std()
                # norm_cond = (cond) / cond.std()
                # TODO: check and modify the shape of the data and cond
                # print(norm_data.shape)
                record['data'] = norm_data.transpose(0, 1)
                record['cond'] = cond.reshape(3, 2)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': cond,
            }
        elif task_id == 5: 
            for record in minibatch:
                data = record['data']
                cond = record['cond']
                # print(f'data.shape:{data.shape}')
                # TODO: move out of the loop
                # norm_data = (data - data.mean()) / data.std()
                # norm_cond = (cond) / cond.std()
                # TODO: check and modify the shape of the data and cond
                # print(norm_data.shape)
                record['data'] = data
                record['cond'] = cond
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': cond,
            }
        
        elif task_id in [6, 7, 8, 9]:
            for record in minibatch:
                # Filter out records that aren't long enough.
                down_sample = torch.view_as_real(record['data']).permute(1, 2, 0)
                up_sample = torch.view_as_real(record['cond']).permute(1, 2, 0)
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                norm_cond = (up_sample - up_sample.mean()) / up_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
                record['cond'] = norm_cond.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': cond.squeeze(2),
            }
        else:
            raise ValueError("Unexpected task_id.")


def from_path(params, is_distributed=False):
    data_dir = params.data_dir
    task_id = params.task_id
    data_key = params.data_key
    cond_key = params.cond_key
    if task_id == 0:
        dataset = WiFiDataset(data_dir, data_key, cond_key)
    elif task_id == 1:
        dataset = FMCWDataset(data_dir)
    elif task_id == 2:
        dataset = MIMODataset(data_dir)
    elif task_id == 3:
        dataset = EEGDataset(data_dir)
    elif task_id == 4:
        dataset = coordCSIDataset(data_dir)
    elif task_id == 5:
        dataset = diffusionDataset(data_dir)
    elif task_id in [6, 7, 9]:
        dataset = exp_2_12Dataset(data_dir)
    elif task_id == 8:
        dataset = exp_2_16Dataset(data_dir, params.vae_path)
    elif task_id == 10:
        dataset = exp_3_7_Argos(data_dir, params.lora, params.FT_id, params.seen_idx, params.ant_idx)
    elif task_id == 11:
        dataset = exp_4_3_multiband(data_dir)
    elif task_id == 12:
        dataset = exp_vivo(data_dir, params.lora, params.FT_id, params.seen_idx, params.ant_idx)
    else:
        raise ValueError("Unexpected task_id.")
    if task_id in [5, 10]:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            collate_fn=Collator(params).collate,
            shuffle=not is_distributed,
            # num_workers=8,
            sampler=DistributedSampler(dataset) if is_distributed else None,
            # pin_memory=True,
            # drop_last=True,
            # persistent_workers=True
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            collate_fn=Collator(params).collate,
            shuffle=not is_distributed,
            num_workers=8,
            sampler=DistributedSampler(dataset) if is_distributed else None,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True)


def from_path_inference(params):
    cond_dir = params.cond_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(cond_dir, params.data_key, params.cond_key)
    elif task_id == 1:
        dataset = FMCWDataset(cond_dir)
    elif task_id == 2:
        dataset = MIMODataset(cond_dir)
    elif task_id == 3:
        dataset = EEGDataset(cond_dir)
    elif task_id == 4:
        dataset = coordCSIDataset(cond_dir)
    elif task_id == 5:
        dataset = diffusionDataset(cond_dir)
    elif task_id in [6, 7, 9]:
        dataset = exp_2_12Dataset(cond_dir)
    elif task_id == 8:
        dataset = exp_2_16Dataset(cond_dir, params.vae_path)
    elif task_id == 10:
        dataset = exp_3_7_Argos(cond_dir, params.lora, params.FT_id, params.seen_idx, params.ant_idx)
    elif task_id == 11:
        dataset = exp_4_3_multiband(cond_dir)
    elif task_id == 12:
        dataset = exp_vivo(cond_dir, params.lora, params.FT_id, params.seen_idx, params.ant_idx)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        collate_fn=Collator(params).collate,
        shuffle=True,
        num_workers=os.cpu_count()
        )
