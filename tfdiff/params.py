import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

# ========================
# Wifi Parameter Setting.
# ========================
params_wifi = AttrDict(
    task_id=0,
    log_dir='./log/wifi',
    model_dir='./model/wifi/b32-256-100s',
    data_dir=['./dataset/wifi/raw'],
    out_dir='./dataset/wifi/output',
    cond_dir=['./dataset/wifi/cond'],
    fid_pred_dir = './dataset/wifi/img_matric/pred',
    fid_data_dir = './dataset/wifi/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    input_dim=90,
    extra_dim=[90],
    cond_dim=6,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)

# ========================
# FMCW Parameter Setting.
# ========================
params_fmcw = AttrDict(
    task_id=1,
    log_dir='./log/fmcw',
    model_dir='./model/fmcw/b32-256-100s',
    data_dir=['./dataset/fmcw/raw'],
    out_dir='./dataset/fmcw/output',
    cond_dir=['./dataset/fmcw/cond'],
    fid_pred_dir = './dataset/fmcw/img_matric/pred',
    fid_data_dir = './dataset/fmcw/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=32,
    learning_rate=1e-3,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    input_dim=128,
    extra_dim=[128],
    cond_dim=6,
    # Model params
    embed_dim=256,
    hidden_dim=256,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
)

# =======================
# MIMO Parameter Setting.
# =======================
params_mimo = AttrDict(
    task_id=2,
    log_dir='./log/mimo',
    model_dir='./model/mimo/b32-256-200s',
    data_dir=['./dataset/mimo/cond'],
    out_dir='./dataset/mimo/output',
    cond_dir=['./dataset/mimo/cond'],
    # Training params
    max_iter=None, # Unlimited number of iterations.
    # for inference use
    batch_size = 8,
    # batch_size=24,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=14,
    # TransEmbedding
    extra_dim=[26, 96],
    cond_dim= [26, 96],
    # Model params
    embed_dim=256,
    spatial_hidden_dim=128,
    tf_hidden_dim=256,
    num_heads=8,
    num_spatial_block=16,
    num_tf_block=16,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)


# ======================
# EEG Parameter Setting. 
# ======================
params_eeg = AttrDict(
    task_id=3,
    log_dir='./log/eeg',
    model_dir='./model/eeg/b32-256-200s',
    data_dir=['./dataset/eeg/raw'],
    out_dir='./dataset/eeg/output',
    cond_dir=['./dataset/eeg/cond'],
    # Training params
    max_iter=None, # Unlimited number of iterations.
    # for inference use
    batch_size = 8,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    extra_dim=[1,1], 
    cond_dim=512,   
    # Model params
    embed_dim=256,
    hidden_dim=256,
    input_dim=1,
    num_block=16,
    num_heads=8,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)

# ========================
# coordCSI Parameter Setting.
# ========================
params_coord2CSI = AttrDict(
    task_id=4,
    log_dir='./log/coordCSI/exp_10_17',
    model_dir='./model/coordCSI/exp_10_17',
    data_dir='/home/zuser/project/WQ/bedroom/camera_1D_3311/mat',
    out_dir='./output/coordCSI/exp_10_17', 
    cond_dir='/home/zuser/project/WQ/bedroom/camera_1D_3311/mat',
    fid_pred_dir = './dataset/wifi/img_matric/pred',
    fid_data_dir = './dataset/wifi/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=32,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    # TODO:change parameters below
    sample_rate=200,
    input_dim=54,
    extra_dim=[54],
    cond_dim=3,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
    # ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)

params_synthesis = AttrDict(
    task_id=5,
    device = 'cuda:1',
    log_dir='./log/synthesis/exp_10_18',
    data_dir = '/media/zuser/Harddisk4TB/WQ/SANDiff/dataset/set5/train_data.npz',
    model_dir = './model/synthesis/exp_10_18',
    out_dir='./inference/output/exp_10_18', 
    cond_dir = '/media/zuser/Harddisk4TB/WQ/SANDiff/dataset/set5/test_data.npz',
    fid_pred_dir = './dataset/wifi/img_matric/pred',
    fid_data_dir = './dataset/wifi/img_matric/data',
    # Training params
    max_iter = None, # Unlimited number of iterations.
    batch_size = 8,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=32,
    robust_sampling=True,
    # Data params
    # TODO:change parameters below
    sample_rate= 200,
    input_dim=216,
    extra_dim=[216],
    cond_dim=6,
    # Model params
    embed_dim=256,
    hidden_dim=256,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    # TODO:whether to use signal_diffusion
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)

params_exp_2_12 = AttrDict(
    task_id=6,
    log_dir='./log/exp_2_12',
    model_dir='./model/exp_2_12',
    data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0'],
    out_dir='./dataset/exp_2_12/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1'], # test set
    fid_pred_dir = './dataset/exp_2_12/img_matric/pred',
    fid_data_dir = './dataset/exp_2_12/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=108,
    input_dim=1,
    extra_dim=[1],
    cond_dim=108,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)

params_exp_federated_2_13 = AttrDict(
    task_id=7,
    log_dir='./log/exp_fed_2_13',
    model_dir='./model/exp_fed_2_13',
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0'],
    client_data_dirs=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region3',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region4',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6',
                      ],
    data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
                    '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
                    '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
                    '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region3',
                    '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region4',
                    '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
                    '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6',
                    ],
    out_dir='./dataset/exp_fed_2_13/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9'], # test set
    fid_pred_dir = './dataset/exp_2_12/img_matric/pred',
    fid_data_dir = './dataset/exp_2_12/img_matric/data',
    # Federated params
    fed_rounds=100,
    local_epochs=1,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=108,
    input_dim=1,
    extra_dim=[1],
    cond_dim=108,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=16,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)

params_exp_federated_2_14 = AttrDict(
    task_id=7,
    log_dir='./log/exp_fed_2_14',
    model_dir='./model/exp_fed_2_14',
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0'],
    client_data_dirs=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region3',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region4',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6',
                      ],
    out_dir='./dataset/exp_fed_2_14/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9'], # test set
    fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
    fid_data_dir = './dataset/exp_2_14/img_matric/data',
    # Federated params
    fed_rounds=150,
    local_epochs=1,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=108,
    input_dim=1,
    extra_dim=[1],
    cond_dim=108,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)

params_exp_VaeFederated_2_16 = AttrDict(
    task_id=8,
    log_dir='./log/exp_FedVae_2_16',
    model_dir='./model/exp_FedVae_2_16',
    vae_path = '/media/zuser/Harddisk4TB/WQ/SANDiff/model/vae/exp_2_15/vae_model_4566000.pth',
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0'],
    client_data_dirs=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region3',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region4',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
                      '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6',
                      ],
    out_dir='./dataset/exp_FedVae_2_16/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9'], # test set
    fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
    fid_data_dir = './dataset/exp_2_14/img_matric/data',
    # Federated params
    fed_rounds=200,
    local_epochs=1,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=26,
    input_dim=1,
    extra_dim=[1],
    cond_dim=26,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)


params_exp_Vae_2_17 = AttrDict(
    task_id=8,
    log_dir='./log/exp_Vae_2_18_2',
    model_dir='./model/exp_Vae_2_18_2',
    vae_path = '/media/zuser/Harddisk4TB/WQ/SANDiff/model/vae/exp_2_15/vae_model_4566000.pth',
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0'],
    data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
                    #   '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
                    #   '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
                    #   '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region3',
                    #   '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region4',
                    #   '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
                    #   '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6',
                      ],
    out_dir='./dataset/exp_Vae_2_18_2/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9'], # test set
    fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
    fid_data_dir = './dataset/exp_2_14/img_matric/data',
    # Federated params
    fed_rounds=200,
    local_epochs=1,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=26,
    input_dim=1,
    extra_dim=[1],
    cond_dim=108,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=16,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)


params_exp_LoRA_2_27 = AttrDict(
    task_id=9,
    lora=True,
    log_dir='./log/exp_LoRA_3_4/FT_region6',
    model_dir='./model/exp_LoRA_3_4/FT_region6',
    data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6'],
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region7',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region8',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9',
            # ],
    out_dir='./dataset/exp_LoRA_3_4/FT_region6/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region6'], # test set
    # fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
    # fid_data_dir = './dataset/exp_2_14/img_matric/data',
    # Federated params
    # fed_rounds=150,
    # local_epochs=1,
    # LoRA params
    lora_r = 8,
    lora_alpha = 1,
    lora_dropout = 0.,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # Data params
    sample_rate=108,
    input_dim=1,
    extra_dim=[1],
    cond_dim=108,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    #ADD:data_key, cond_key
    data_key='feature',
    cond_key='cond',
)

# params_exp_Argos_3_7 = AttrDict(
#     task_id=10,
#     lora=False,
#     seen_idx = [2],
#     # seen_idx = [0, 1, 3, 4, 5],
#     FT_id = 0,
#     ant_idx= 2,
#     log_dir='./log/exp_Argos_4_27',
#     model_dir='./model/exp_Argos_4_27',
#     data_dir=['/media/zuser/Harddisk4TB/WQ/Raw-CSI-Data/5026time_6client_96ant_52sub.npy'],
#     # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
#     #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
#     #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
#     #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region7',
#     #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region8',
#     #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
#     #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9',
#             # ],
#     out_dir='./dataset/exp_Argos_4_27/output',
#     cond_dir=['/media/zuser/Harddisk4TB/WQ/Raw-CSI-Data/5026time_6client_96ant_52sub.npy'], # test set
#     # fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
#     # fid_data_dir = './dataset/exp_2_14/img_matric/data',
#     # Federated params
#     # fed_rounds=150,
#     # local_epochs=1,
#     # LoRA params
#     lora_r = 8,
#     lora_alpha = 1,
#     lora_dropout = 0.,
#     # Training params
#     max_iter=None, # Unlimited number of iterations.
#     batch_size=8,
#     learning_rate=1e-4,
#     max_grad_norm=None,
#     # Inference params
#     inference_batch_size=16,
#     robust_sampling=True,
#     sample_rate=14,
#     # TransEmbedding
#     extra_dim=[26, 1],
#     cond_dim= [26, 1],
#     # Model params
#     embed_dim=256,
#     spatial_hidden_dim=128,
#     tf_hidden_dim=256,
#     num_heads=8,
#     num_spatial_block=16,
#     num_tf_block=16,
#     dropout=0.,
#     mlp_ratio=4,
#     learn_tfdiff=False,
#     # Diffusion params
#     signal_diffusion=True,
#     max_step=200,
#     # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
#     blur_schedule=((0.1**2) * np.ones(200)).tolist(),
#     # \beta_t, noise level added to the signal on each diffusion step [T]
#     noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
# )

params_exp_Argos_3_7 = AttrDict(
    task_id=10,
    lora=True,
    # seen_idx = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
    seen_idx = [-1],
    FT_id = 4,
    ant_idx= -1,
    log_dir='./log/exp_Argos_4_30/FT_Client4',
    # When fine-tuning, remember to move base model to model_dir
    model_dir='./model/exp_Argos_4_30/FT_Client4',
    data_dir=['/mnt/data/WQ/Raw-CSI-Data/set2_4_29/5026_6_96_52_train_data.npy'],
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region7',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region8',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9',
            # ],
    out_dir='./inference/exp_Argos_4_30/FT4_Wasserstein_L2_Client4',
    cond_dir=['/mnt/data/WQ/Raw-CSI-Data/set2_4_29/5026_6_96_52_test_data2.npy'], # test set
    # fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
    # fid_data_dir = './dataset/exp_2_14/img_matric/data',
    # Federated params
    # fed_rounds=150,
    # local_epochs=1,
    # LoRA params
    lora_r = 8,
    lora_alpha = 1,
    lora_dropout = 0.,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # sample rate is fixed by generatiing dataset
    sample_rate=14,
    # TransEmbedding
    extra_dim=[26, 1],
    cond_dim= [26, 1],
    # Model params
    embed_dim=128,
    spatial_hidden_dim=128,
    tf_hidden_dim=128,
    num_heads=8,
    num_spatial_block=12,
    num_tf_block=12,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)

params_exp_4_3_multiband = AttrDict(
    lora = False,
    task_id=11,
    log_dir='./log/exp_4_7_2_multibandZHR',
    model_dir='./model/exp_4_7_2_multibandZHR',
    data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/multi_band_ZHR/set2'],
    out_dir='./dataset/exp_4_7_2_multibandZHR/output',
    cond_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/multi_band_ZHR/set2'],
    # Training params
    max_iter=None, # Unlimited number of iterations.
    # for inference use
    batch_size = 16,
    # batch_size=24,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=64,
    robust_sampling=True,
    # Data params
    sample_rate=1,
    # TransEmbedding
    extra_dim=[20, 5],
    cond_dim= [20, 1],
    # Model params
    embed_dim=256,
    spatial_hidden_dim=128,
    tf_hidden_dim=256,
    num_heads=8,
    num_spatial_block=16,
    num_tf_block=16,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)

params_exp_vivo_9_15 = AttrDict(
    task_id=12,
    lora=True,
    seen_idx = [-1],
    # seen_idx = [i for i in range(15)],
    FT_id = 14,
    ant_idx= -1,
    log_dir='./log/exp_vivo_10_5/FT14',
    # When fine-tuning, remember to move base model to model_dir
    model_dir='./model/exp_vivo_10_5/FT14',
    data_dir=['/mnt/data/WQ/Raw-CSI-Data/vivo1_9_17/15680_21_5_32_train_data.npy'],
    # data_dir=['/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region0', 
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region1', 
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region2',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region7',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region8',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region5',
    #             '/media/zuser/Harddisk4TB/WQ/bedroom/exp_2_12_FedDiff/exp_2_12_1/region9',
            # ],
    out_dir='./inference/exp_vivo_10_5/FT14',
    cond_dir=['/mnt/data/WQ/Raw-CSI-Data/vivo1_9_17/3940_21_5_32_test_data.npy'], # test set
    # cond_dir=['/mnt/data/WQ/Raw-CSI-Data/vivo1_9_17/15680_21_5_32_train_data.npy'], # test set
    # fid_pred_dir = './dataset/exp_2_14/img_matric/pred',
    # fid_data_dir = './dataset/exp_2_14/img_matric/data',
    # Federated params
    # fed_rounds=150,
    # local_epochs=1,
    # LoRA params
    lora_r = 8,
    lora_alpha = 1,
    lora_dropout = 0.,
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=16,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=16,
    robust_sampling=True,
    # sample rate is fixed by generatiing dataset
    sample_rate=10,
    # TransEmbedding
    extra_dim=[1, 32],
    cond_dim= [1, 32],
    # Model params
    embed_dim=256,
    spatial_hidden_dim=128,
    tf_hidden_dim=256,
    num_heads=8,
    num_spatial_block=16,
    num_tf_block=16,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)

all_params = [params_wifi, params_fmcw, params_mimo, params_eeg, params_coord2CSI, params_synthesis, params_exp_2_12,
               params_exp_federated_2_13, params_exp_Vae_2_17, params_exp_LoRA_2_27, params_exp_Argos_3_7, params_exp_4_3_multiband, params_exp_vivo_9_15]