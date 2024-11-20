import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import fsspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass 
class TrainerConfig:
    checkpoint_path: Optional[str]=None
    batch_size: int=None
    num_loaders: int =None
    use_amp: bool=None
    save_every: int=None
    max_norms: float=None
    max_epochs: int=None
    
@dataclass    
class Checkpoint:
    model_state: "OrderedDict[str, torch.tensor]"
    optimizer_state: Dict[str,Any]
    finished_epoch: int
    
    
class Trainer:
    def __init__(self,config: TrainerConfig,model,optimizer,train_data,val_data=None):
        self.local_rank=int(os.environ["LOCAL_RANK"])
        self.global_rank=int(os.environ["RANk"])
        self.config=config
        self.model=model.to(self.local_rank)
        self.train_data=self._process_dataloader(train_data)
        self.val_data=self._process_dataloader(val_data)
        self.optimizer=optimizer
        self.epochs_run=0
        if self.config.use_amp:
            self.grad_scaler=torch.cuda.amp.GradScaler()
        if self.config.checkpoint_path is None:
            self.config.checkpoint_path="checkpoint.pt"
        self.save_every=self.config.save_every    
        self._load_checkpoint()
        

    def _process_dataloader(self,data: Dataset):
        return DataLoader(data,batch_size=self.config.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=self.config.num_loaders,
                          sampler=DistributedSampler(data))
    
    def _load_checkpoint(self):
        try:
            checkpoint=fsspec.open(self.config.checkpoint_path)
            with checkpoint as f:
                checkpoint_data=torch.load(f,map_location="cpu")
        except FileNotFoundError:
            print("not found file")
            return
        checkpoint=Checkpoint(**checkpoint_data)
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.epochs_run=checkpoint.finished_epoch
    
    def _save_checkpoint(self,epoch):
        model=self.model
        raw_model=model.module if hasattr(model,"module") else model
        checkpoint=Checkpoint(model_state=raw_model,
                              optimizer_state=self.optimizer,
                              finished_epoch=epoch)
        
        checkpoint=asdict(checkpoint)
        torch.save(checkpoint,self.config.checkpoint_path)
        print(f"saved at {epoch}")
        
        
    
    def _run_batch(self,source,targets,train: bool=True) ->float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(self.config.use_amp)):
            _,loss=self.model(source,targets)
        if train:
            self.grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.config.max_norms)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.config.max_norms)
            self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self,epoch,dataloader: DataLoader, train: bool=True):
        dataloader.sampler.set_epoch(epoch)
        for iter,(source,targets) in enumerate(dataloader):
            state="train" if train else "val"
            source=source.to(self.local_rank)
            targets=targets.to(self.local_rank)
            loss=self._run_batch(source,targets)
            if iter%100==0:
                print(f"[GPU {self.local_rank}] | loss {loss} | Epoch {epoch} | state {state}")
    
    def train(self):
        for epoch in range(self.epochs_run,self.config.max_epochs):
            
            self._run_epoch(epoch,self.train_data,train=True)
            
            if self.local_rank==0 and epoch%self.config.save_every==0:
                self._save_checkpoint(epoch)
            if self.val_data:
                self._run_epoch(epoch,self.val_data,train=False)
                
    
                
                
    