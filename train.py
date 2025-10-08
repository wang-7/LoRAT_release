import os

import torch
from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel

from argparse import ArgumentParser

from tfdiff.params import all_params
from tfdiff.learner import tfdiffLearner
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.mimo_lora import tfdiff_mimo_LoRA
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.coord2CSI_model import tfdiff_coord2CSI
from tfdiff.wifi_lora import tfdiff_WiFi_LoRA
from tfdiff.dataset import from_path
from tfdiff.federator import FederatedLearner
import json

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def _train_impl(replica_id, model, dataset, params):
    opt = torch.optim.AdamW(model.parameters(), lr=params.lora_learning_rate if params.lora else params.learning_rate)
    learner = tfdiffLearner(params.log_dir, params.model_dir, model, dataset, opt, params)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_iter=params.max_iter)

def choose_model(params):
    if params.task_id==0:
        model = tfdiff_eeg(params).cuda()
    elif params.task_id in [1, 10, 12]:
        if params.lora:
            model = tfdiff_mimo_LoRA(params).cuda()
        else:
            model = tfdiff_mimo(params).cuda()
    elif params.task_id==2:
        model = tfdiff_WiFi(params).cuda()
    elif params.task_id==4:
        model = tfdiff_coord2CSI(params).cuda()
    elif params.task_id==5:
        model = tfdiff_coord2CSI(params).cuda()
    elif params.task_id in [6, 7, 8]:
        model = tfdiff_WiFi(params).cuda()
    elif params.task_id==9:
        if params.lora:
            model = tfdiff_WiFi_LoRA(params).cuda()
        else:
            model = tfdiff_WiFi(params).cuda()
    elif params.task_id==11:
        model = tfdiff_mimo(params).cuda()
    else:
        raise ValueError("Unexpected task_id.")
    return model

def train(params):
    dataset = from_path(params)
    model = choose_model(params)
    _train_impl(0, model, dataset, params)

def train_federated(params, client_data_dirs, num_rounds=10, local_epochs=1):
    # Initialize federated learner
    fed_learner = FederatedLearner(params, client_data_dirs)
    
    # load global model
    fed_learner.load_global_model()

    # Start federated training
    fed_learner.train(num_rounds=num_rounds, local_epochs=local_epochs)
    
    # Save final global model
    fed_learner.save_global_model()

def train_distributed(replica_id, replica_count, port, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=replica_id, world_size=replica_count)
    dataset = from_path(params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = choose_model(params)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, params)


def main(args):
    torch.manual_seed(11)
    params = all_params[args.task_id]
    params_dict = {k: v for k, v in vars(params).items() if not k.startswith('_')}
    os.makedirs(params.model_dir, exist_ok=True)
    with open(os.path.join(params.model_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)
    params.data_key = args.data_key or params.data_key
    params.cond_key = args.cond_key or params.cond_key
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.model_dir is not None:
        params.model_dir = args.model_dir
    if args.data_dir is not None:
        params.data_dir = args.data_dir
    if args.log_dir is not None:
        params.log_dir = args.log_dir
    if args.max_iter is not None:
        params.max_iter = args.max_iter
    replica_count = device_count()
    if replica_count > 1:
        if params.batch_size % replica_count != 0:
            raise ValueError(
                f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, params), nprocs=replica_count, join=True)
    elif args.federated:
        train_federated(
            params=params,
            client_data_dirs=params.client_data_dirs,  # List of client data directories
            num_rounds=params.fed_rounds,
            local_epochs=params.local_epochs
        )
    else:
        train(params)


# python train.py --task_id [task_id] --model_dir [model_dir] --data_dir [data_dir]
# HF_ENV_NAME=py38-202207 hfai python train.py --task_id [task_id] --model_dir [model_dir] --data_dir [data_dir] --max_iter [iter_num] --batch_size [batch_size] -- -n [node_num] --force
if __name__ == '__main__':
    parser = ArgumentParser(
        description='train (or resume training) a tfdiff model')
    parser.add_argument('--task_id', type=int,
                        help='use case of tfdiff model, 0/1/2/3 for WiFi/FMCW/MIMO/EEG respectively')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dir', default=None, nargs='+',
                        help='space separated list of directories from which to read csi files for training')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--max_iter', default=None, type=int,
                        help='maximum number of training iteration')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--data_key', default='feature', help='specify the key of data in .mat dict')
    parser.add_argument('--cond_key', default='cond', help='specify the key of cond in .mat dict')
    parser.add_argument('--federated', default=False, type=bool, help='enable federated training')
    main(parser.parse_args())
