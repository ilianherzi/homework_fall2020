from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import torch.nn.functional as F
    

class MomentumCountModel(nn.Module, BaseExplorationModel):
    '''MomentumCountModel idea: keep track of the internal representation associated with each observation and predicted action and
       take a moving average of those internal representations
    '''
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['cm_output_size'] # should be the number of actions
        self.n_layers = hparams['cm_n_layers']
        self.size = hparams['cm_size']
        self.optimizer_spec = optimizer_spec
        self.alpha = hparams['cm_alpha']
         
        self.encoder = ptu.build_mlp(
            input_size=self.ob_dim, # b/c will also take as input it's own internal representation
            output_size=self.size, 
            n_layers=self.n_layers,
            size=self.size, 
        )
        self.f = ptu.build_mlp(
            input_size=self.size,
            output_size=self.output_size,
            n_layers=1,
            size=self.size
        )
        
        self.optimizer = self.optimizer_spec.constructor(
            list(self.encoder.parameters()) + list(self.f.parameters()),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.encoder(ptu.device)
        self.f.to(ptu.device)
        self.memory = {}
    
    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)
        
    def forward(self, ob_no):
        acs_representation = (self.encoder(ob_no))
        key_batch = int(((acs_representation.sign() + 1)/2).sum(dim=1)) # sum across the columns
        rewards = []
        for key in key_batch:
            if key not in self.memory:
                self.memory[key] = acs_representation / acs_representation.norm()
                rewards.append(self.memory[key].norm())
            else:
                rewards.append(((acs_representation) / acs_representation.norm()).T @ self.memory[key])
                self.memory[key] = (1 - self.alpha) * self.memory[key] + self.alpha * acs_representation
                self.memory[key] = self.memory[key] / self.memory[key].norm()
        return rewards
    
    def update(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        loss = (self(ob_no) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
        
        
        

