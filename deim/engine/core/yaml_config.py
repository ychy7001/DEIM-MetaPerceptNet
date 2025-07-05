"""   
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)  
Copyright(c) 2023 lyuwenyu. All Rights Reserved.   
"""

import torch  
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader

import re 
import copy

from ._config import BaseConfig
from .workspace import create 
from .yaml_utils import load_config, merge_config, merge_dict
     
class YAMLConfig(BaseConfig):  
    def __init__(self, cfg_path: str, **kwargs) -> None:    
        super().__init__()
 
        cfg = load_config(cfg_path) 
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)     
    
        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]     
 
    @property 
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)   
  
    @property  
    def model(self, ) -> torch.nn.Module:  
        if self._model is None and 'model' in self.yaml_cfg: 
            self._model = create(self.yaml_cfg['model'], self.global_cfg)     
        return super().model    
   
    @property
    def postprocessor(self, ) -> torch.nn.Module:  
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)     
        return super().postprocessor 
     
    @property    
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)   
        return super().criterion

    @property     
    def optimizer(self, ) -> optim.Optimizer:     
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)     
            self._optimizer = create('optimizer', self.global_cfg, params=params)    
        return super().optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:     
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer) 
            print(f'Initial lr: {self._lr_scheduler.get_last_lr()}')   
        return super().lr_scheduler
   
    @property     
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :  
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler) 
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader: 
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')     
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:  
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader     

    @property
    def ema(self, ) -> torch.nn.Module:  
        if self._ema is None and self.yaml_cfg.get('use_ema', False):    
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema

    @property  
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):   
            self._scaler = create('scaler', self.global_cfg)    
        return super().scaler

    @property   
    def evaluator(self, ):     
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)    
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)     
            else:   
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")  
        return super().evaluator     
     
    @staticmethod    
    def get_optim_params(cfg: dict, model: nn.Module):
        """  
        E.g.:  
            ^(?=.*a)(?=.*b).*$  means including a and b    
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b  
        """   
        assert 'type' in cfg, 'Optimizer config must have a "type" field.' 
        cfg = copy.deepcopy(cfg)   

        # If no custom params are defined, return all model parameters directly
        if 'params' not in cfg:
            names = [k for k, v in model.named_parameters() if v.requires_grad]
            print(f"Total trainable parameters names: {len(names)}")
            print(f"Visited parameters names (unique): {len(names)}") # All visited by default
            return model.parameters() 
  
        assert isinstance(cfg['params'], list), 'Optimizer "params" field must be a list.'

        param_groups = []
        # Use a set to track names of parameters that have been assigned to a group.
        # This prevents duplicates and ensures each parameter belongs to only one group.
        processed_names_set = set() 

        all_trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        
        # 1. Process explicit parameter groups defined in the config
        for pg_cfg in cfg['params']:     
            pattern = pg_cfg['params']  # The regex pattern for this group
            current_group_params_values = []
            
            # Iterate through all trainable parameters to find matches for the current pattern
            for name, param in all_trainable_params.items():
                if name in processed_names_set:
                    # This parameter has already been assigned to a previous explicit group, skip it.
                    continue
                
                # Check if the parameter name matches the current regex pattern
                if re.findall(pattern, name):
                    current_group_params_values.append(param)
                    processed_names_set.add(name) # Mark as processed
            
            # If this group actually collected any parameters, add it to param_groups
            if current_group_params_values:
                # Create a new dictionary for this group, copying other properties like lr, weight_decay
                group_dict = {k: v for k, v in pg_cfg.items() if k != 'params'}
                group_dict['params'] = current_group_params_values
                param_groups.append(group_dict)
        
        # 2. Handle any remaining parameters not covered by explicit groups (the "default" group)
        unseen_params_values = []
        for name, param in all_trainable_params.items():
            if name not in processed_names_set:
                unseen_params_values.append(param)
                processed_names_set.add(name) # Mark as processed for the default group
        
        if unseen_params_values:
            # Create a default group for these parameters
            default_group_dict = {'params': unseen_params_values}
            # Add default lr, weight_decay from the main optimizer config if they exist
            if 'lr' in cfg:
                default_group_dict['lr'] = cfg['lr']
            if 'weight_decay' in cfg:
                default_group_dict['weight_decay'] = cfg['weight_decay']
            
            param_groups.append(default_group_dict)

        # 3. Final Assertion: Verify all trainable parameters have been uniquely assigned.
        names_of_all_trainable_params = list(all_trainable_params.keys()) # Get a list of all trainable param names

        print(f"Total trainable parameters names: {len(names_of_all_trainable_params)}")
        print(f"Visited parameters names (unique): {len(processed_names_set)}")
        
        # Identify any parameters that were not processed (should be empty)
        unvisited = set(names_of_all_trainable_params) - processed_names_set
        if unvisited:
            print(f"ERROR: Unvisited parameters after grouping: {unvisited}")
        
        # This check is mostly for debugging the logic itself, if processed_names_set somehow got extra names
        extra_visited = processed_names_set - set(names_of_all_trainable_params)
        if extra_visited:
            print(f"ERROR: Parameters visited but not in total trainable params (logic error?): {extra_visited}")

        # The core assertion: ensures every trainable parameter is uniquely visited.
        assert len(processed_names_set) == len(names_of_all_trainable_params), \
            f'Assertion failed: Expected {len(names_of_all_trainable_params)} unique trainable params, ' \
            f'but {len(processed_names_set)} were processed. ' \
            f'Unvisited: {unvisited}. Extra/duplicate visited: {extra_visited}'

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):    
        """compute batch size for per rank if total_batch_size is provided.     
        """ 
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'
   
        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size() 
        return bs  

    def build_dataloader(self, name: str):    
        bs = self.get_rank_batch_size(self.yaml_cfg[name])    
        global_cfg = self.global_cfg 
        if 'total_batch_size' in global_cfg[name]:     
            # pop unexpected key for dataloader init 
            _ = global_cfg[name].pop('total_batch_size')
        print(f'building {name} with batch_size={bs}...') 
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)    
        return loader  