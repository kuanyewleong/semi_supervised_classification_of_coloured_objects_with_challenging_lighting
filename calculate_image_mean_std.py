from typing import Optional
import argparse
from pathlib import Path
import random
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.utils.data.dataloader import DataLoader

from color_recognition.datasets import CRDatasetCollection, DatasetType, FetchMode



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path, help='toml path of train and validation datasets')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for validation (default: 1000)')    
    parser.add_argument('--skip_confirm', action='store_true', default=False, help='disables confirmation')
    parser.add_argument('--fetch-mode',
                        help='Fetch mode',
                        type=str,
                        choices=[m.name.lower() for m in FetchMode],
                        default='eager')    
    return parser

def main():
    args = get_parser().parse_args()    
    run(args)


def run(args) -> Optional[int]:
    fetch_mode = FetchMode[args.fetch_mode.upper()]

    dataset_collection = CRDatasetCollection()
    dataset_collection.load_toml(args.dataset, [DatasetType.TRAIN, DatasetType.VALID], fetch_mode)
    dataset_collection.print_summary()

    train_loader = dataset_collection.get_loader(DatasetType.TRAIN,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=4
                                                 )
    assert train_loader is not None
    valid_loader = dataset_collection.get_loader(DatasetType.VALID,
                                                batch_size=args.valid_batch_size,
                                                shuffle=False,
                                                num_workers=4
                                                )
    assert valid_loader is not None

    ####### COMPUTE MEAN / STD
    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    
    # loop through images
    print("Summing train_loader ... ")
    for batch_idx, (data, target, _) in enumerate(train_loader):                
        psum    += data.sum(axis        = [0, 2, 3])
        psum_sq += (data ** 2).sum(axis = [0, 2, 3])
    
    print("Summing valid_loader ... ")
    for batch_idx, (data, target, _) in enumerate(valid_loader):                
        psum    += data.sum(axis        = [0, 2, 3])
        psum_sq += (data ** 2).sum(axis = [0, 2, 3])
            
    # pixel count
    total_samples = len(train_loader.dataset) + len(valid_loader.dataset)
    count = total_samples * data.shape[2] * data.shape[3]

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print("Total samples: ", total_samples)
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

if __name__ == '__main__':
    main()


