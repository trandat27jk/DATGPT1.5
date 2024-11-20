import torch
from dataclasses import dataclass
import tiktoken
from torch.utils.data import DataLoader,Dataset
@dataclass
class DataConfig:
    block_size: int =None
    encoding: str=None
    stride: int=None
    

class StoryTellingDataset(Dataset):
    def __init__(self,txt,config: DataConfig):
        super().__init__()
        self.inputs=[]
        self.targets=[]
        for i in range(0,len(txt)-config.block_size,config.stride):
            inputs=txt[i:i+config.block_size]
            targets=txt[i+1:i+1+config.block_size]
            self.inputs.append(inputs)
            self.targets.append(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index],self.targets[index]         
        