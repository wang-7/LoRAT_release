import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import _nested_map
import loralib as lora

def toimage(features, pred):
    d_sample = torch.view_as_complex(features['data'][0]).abs()
    p_sample = torch.view_as_real(pred).abs()
    sample_grid = torchvision.utils.make_grid([d_sample, p_sample], padding=8)[0].to('cpu')
    # 创建一个新的figure对象
    fig, ax = plt.subplots()
    # 使用imshow显示热力图
    im = ax.imshow(sample_grid)
    # 将figure保存到一个BytesIO缓冲区中
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=False)
    buf.seek(0)
    # 从缓冲区读取数据并解码为numpy数组
    image = plt.imread(buf)[:, :, :3]
    # 关闭figure
    plt.close(fig)
    return image

class tfdiffLoss(nn.Module):
    def __init__(self, w=0.1):
        super().__init__()
        self.w = w

    def forward(self, target, est, target_noise=None, est_noise=None):
        target_fft = torch.fft.fft(target, dim=1) 
        est_fft = torch.fft(est)
        t_loss = self.complex_mse_loss(target, est)
        f_loss = self.complex_mse_loss(target_fft, est_fft)
        n_loss = self.complex_mse_loss(target_noise, est_noise) if (target_noise and est_noise) else 0.
        return (t_loss + f_loss + self.w * n_loss)

    def complex_mse_loss(self, target, est):
        target = torch.view_as_complex(target)
        est = torch.view_as_complex(est)
        return torch.mean(torch.abs(target-est)**2)
        

class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.task_id = params.task_id
        self.log_dir = log_dir
        self.model = model
        if params.lora:
            lora.mark_only_lora_as_trainable(self.model)
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        # self.prof = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=2, active=1, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
        #     with_modules=True, with_flops=True
        # )
        # eeg
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, 5, gamma=0.5)
        # mimo
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 20, gamma=0.5)
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = nn.MSELoss()
        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'], strict=not self.params.lora)
        else:
            missing, unexpected = self.model.load_state_dict(state_dict['model'], strict=not self.params.lora)
            # print(f'Missing keys: {missing}')
            # print(f'Unexpected keys: {unexpected}')
        if not self.params.lora:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                print('islink')
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            print(f'Restored from checkpoint {os.path.abspath(self.model_dir)}/{filename}.pt')
            return True
        except FileNotFoundError:
            print(f'No checkpoint found at {os.path.abspath(self.model_dir)}/{filename}.pt')
            return False


    def train(self, max_iter=None):
        device = next(self.model.parameters()).device
        # self.prof.start()
        while True:  # epoch
            for features in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
                if max_iter is not None and self.iter >= max_iter:
                    # self.prof.stop()
                    return
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_iter(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at iteration {self.iter}.')
                if self.is_master:
                    if self.iter % 1 == 0:
                        self._write_summary(self.iter, features, loss)
                    if self.iter % (10 * len(self.dataset)) == 0:
                        if not self.params.lora:
                            self.save_to_checkpoint()
                        else:
                            checkpoint_path = f'{self.model_dir}/lora'
                            os.makedirs(checkpoint_path, exist_ok=True)
                            torch.save(lora.lora_state_dict(self.model), os.path.join(checkpoint_path, f'weights-{self.iter}.pt'))
                            link_name = f'{checkpoint_path}/lora_weights.pt'
                            if os.path.islink(link_name):
                                os.unlink(link_name)
                            os.symlink(f'weights-{self.iter}.pt', link_name)
                # Update the tqdm postfix with the current loss
                # if self.is_master:
                    # tqdm.write(f'Iteration {self.iter}, Loss: {loss.item()}')
                # self.prof.step()
                self.iter += 1

            # self.lr_scheduler.step()

    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']  # orignial data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64)
        degrade_data = self.diffusion.degrade_fn(
            data, t ,self.task_id)  # degrade data, x_t, [B, N, S*A, 2]
        predicted = self.model(degrade_data, t, cond)
        if self.iter % 1000 == 0:
            save_path = f'{self.log_dir}/intermediate'
            os.makedirs(save_path, exist_ok=True)
            np.savez(f'{save_path}/data_{self.iter}.npz', data=data.detach().cpu().numpy(),
                      degrade_data=degrade_data.detach().cpu().numpy(),
                        predicted=predicted.detach().cpu().numpy(),
                          cond=cond.detach().cpu().numpy(),
                            t=t.detach().cpu().numpy())
        if self.task_id==3:
            data = data.reshape(-1,512,1,2)
        loss = self.loss_fn(data, predicted)
        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()
        return loss

    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        # writer.add_image('train/compare', image, iter)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iter)
        writer.flush()
        self.summary_writer = writer
