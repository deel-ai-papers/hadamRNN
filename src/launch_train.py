# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import os
import sys
import stat

import argparse
try :
    import wandb
except:
    print("No wandb")
    wandb = None
import yaml
import random
import numpy as np
from simple_parsing import ArgumentParser

from training import train, evaluate
from config import Config
from getters import *
from models import *

from utils import find_file



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="conf/permuted_mnist/pmnist_hadamRNN_paper.yaml"
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_network", action="store_true")
    parser.add_argument("--save_name", type=str, default=None)
    return parser.parse_args()


ARGS = parse_args()


def make_model(name, **kwargs):
    Model = get_model(name)
    return Model(**kwargs)


def make_dataset(name, **kwargs):
    Dataset = get_dataset(name)
    return Dataset(**kwargs)


def train_wandb():
    print(ARGS.config)
    conf_file = find_file(ARGS.config)
    config = Config(conf_file=conf_file)
    project_name = config.project
    print(project_name)
    wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,  # "projunn_quantized",
            # track hyperparameters and run metadata
            config=config.config,
        )
    train_local(config, use_wandb=True)
    wandb.finish()

def train_local(config = None, use_wandb = False):
    if config is None:
        print(ARGS.config)
        conf_file = find_file(ARGS.config)
        config = Config(conf_file=conf_file)

    if ARGS.device is not None:
        device = ARGS.device
    else:
        device = config.get_device()

    model = make_model(**config.model).to(device)
    print(model)
    dataset = make_dataset(**config.dataset)

    config.instantiate_optimizer(params=model.parameters())
    config.instantiate_scheduler()

    #train(model, dataset, use_wandb = use_wandb, tqdm_disable=True, **config.train)  #No tqdm for reducing logs on calmip
    train(model, dataset, use_tqdm=True, **config.train)

    if dataset.test_ds is not None:
        test_batch_size = dataset.te_size // 10
        test_ds = dataset.test_ds
    else:
        test_batch_size = dataset.va_size // 10
        test_ds = dataset.val_ds
    stat_test = evaluate(test_ds, test_batch_size, model, loss_fn=config.train['loss_fn'], metrics=config.train['metrics'], kind='test', torch_device=config.train['torch_device'])
    
    if ARGS.save_network:
        if ARGS.save_name is not None:
            run_name = ARGS.save_name
        else:
            run_name = "current_model"
        torch.save(model.state_dict(), f'results/{run_name}.pth')
        # print in file performance
        with open(f'results/{run_name}.txt', 'w') as f:
            for kk in stat_test.keys():
                f.write(f"{kk}: {stat_test[kk]}\n")
                
    if use_wandb:
        for kk in stat_test.keys():
            wandb.run.summary["final test evaluation/"+kk] = stat_test[kk]
    return model


def main():
    if ARGS.use_wandb and wandb is not None:
        train_wandb()
    else:
        train_local()


if __name__ == "__main__":
    main()
