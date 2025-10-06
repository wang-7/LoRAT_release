import copy
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from tfdiff.learner import tfdiffLearner
from tfdiff.dataset import from_path
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.coord2CSI_model import tfdiff_coord2CSI
from tfdiff.params import AttrDict


class FederatedLearner:
    def __init__(self, params, client_data_dirs):
        """
        Initialize the federated learning framework
        
        Args:
            params: Model and training parameters
            client_data_dirs: List of data directories for each client
        """
        self.params = params
        self.client_data_dirs = client_data_dirs
        self.num_clients = len(client_data_dirs)
        self.device = torch.device(params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize global model
        self.global_model = self._create_model()
        self.global_model.to(self.device)

    def _create_model(self):
        """Create model based on task_id""" 
        if self.params.task_id==0:
            model = tfdiff_eeg(self.params).cuda()
        elif self.params.task_id==1:
            model = tfdiff_mimo(self.params).cuda()
        elif self.params.task_id==2:   
            model = tfdiff_WiFi(self.params).cuda()
        elif self.params.task_id==4:
            model = tfdiff_coord2CSI(self.params).cuda()
        elif self.params.task_id==5:
            model = tfdiff_coord2CSI(self.params).cuda()
        elif self.params.task_id in [6, 7, 8]:
            model = tfdiff_WiFi(self.params).cuda()

        else:
            raise ValueError("Unexpected task_id.")
        return model


    def _create_client_learner(self, client_id):
        """Create a learner instance for a client"""
        # Create a deep copy of the global model for this client
        client_model = copy.deepcopy(self.global_model)
        
        # Create new AttrDict with copied parameters
        client_params = AttrDict(copy.deepcopy(dict(self.params)))
        client_params.data_dir = [self.client_data_dirs[client_id]]
        
        # Create dataset for this client
        dataset = from_path(client_params)
        
        # Create optimizer for this client
        optimizer = torch.optim.AdamW(client_model.parameters(), lr=client_params.learning_rate)
        
        # Create learner for this client
        return tfdiffLearner(
            log_dir=f"{client_params.log_dir}/client_{client_id}",
            model_dir=f"{client_params.model_dir}/client_{client_id}",
            model=client_model,
            dataset=dataset,
            optimizer=optimizer,
            params=client_params
        )

    def _aggregate_models(self, client_models):
        """Aggregate client models using FedAvg algorithm"""
        global_state_dict = self.global_model.state_dict()
        
        # Initialize sum dictionary with zeros
        sum_dict = OrderedDict()
        for key in global_state_dict.keys():
            sum_dict[key] = torch.zeros_like(global_state_dict[key])
            
        # Sum up the parameters
        for client_model in client_models:
            client_state_dict = client_model.state_dict()
            for key in global_state_dict.keys():
                sum_dict[key] += client_state_dict[key]
                
        # Average the parameters
        for key in global_state_dict.keys():
            global_state_dict[key] = sum_dict[key] / len(client_models)
            
        # Load the averaged parameters back to global model
        self.global_model.load_state_dict(global_state_dict)

    def train(self, num_rounds, local_epochs):
        """
        Train the model using federated learning
        
        Args:
            num_rounds: Number of federated learning rounds
            local_epochs: Number of local training epochs for each client
        """
        for round in range(num_rounds):
            print(f"\nFederated Learning Round {round + 1}/{num_rounds}")
            
            # Train each client
            client_models = []
            for client_id in range(self.num_clients):
                print(f"\nTraining Client {client_id + 1}/{self.num_clients}")
                
                # Create client learner
                client_learner = self._create_client_learner(client_id)
                
                # Set local training iterations based on epochs
                local_iters = local_epochs * len(client_learner.dataset)
                
                # Train client model
                client_learner.train(max_iter=local_iters)
                
                # Collect trained client model
                client_models.append(client_learner.model)
                
                # Save client model checkpoint
                # client_learner.save_to_checkpoint(f"round_{round+1}")
            
            # Aggregate client models
            self._aggregate_models(client_models)
            
            # Save global model checkpoint
            torch.save(
                self.global_model.state_dict(),
                f"{self.params.model_dir}/global_model_round_{round+1}.pt"
            )

    def save_global_model(self, filename='global_model.pt'):
        """Save the global model"""
        torch.save(self.global_model.state_dict(), f"{self.params.model_dir}/{filename}")

    def load_global_model(self, filename='global_model.pt'):
        """Load the global model"""
        # check if the file exists
        if os.path.exists(f"{self.params.model_dir}/{filename}"):
            self.global_model.load_state_dict(
                torch.load(f"{self.params.model_dir}/{filename}")
            )
            print(f"Loaded global model from {self.params.model_dir}/{filename}")
        else:
            print(f"No global model found in {self.params.model_dir}")