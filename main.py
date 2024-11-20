import os

import hydra
import torch
import torch.nn
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import random_split

from dataloader import DataConfig, StoryTellingDataset
from model import ModelConfig, OptimizerConfig, configure_optimizers, nGPT
from trainer import Trainer, TrainerConfig


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_train_objs(
    modelConfig: ModelConfig, Opt_config: OptimizerConfig, data_config: DataConfig
):
    model = nGPT(modelConfig)
    optimizer = configure_optimizers(model, OptimizerConfig)
    dataset = StoryTellingDataset(DataConfig)
    train_len = int(len(dataset) * data_config.train_split)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    return model, optimizer, train_set, val_set


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    ddp_setup()

    gpt_cfg = ModelConfig(**config["mni"])
