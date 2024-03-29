# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Chest X-ray Shard Descriptor."""

import logging
import os
import torch
import tqdm


from typing import List
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

import torchvision.datasets as datasets
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)


class ChestShardDataset(ShardDataset):
    """Chest Shard dataset class."""

    def __init__(self, x, y, data_type, rank=2, worldsize=2):
        """Initialize CovidDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class ChestShardDescriptor(ShardDescriptor):
    """Chest Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '2, 2',
            **kwargs
    ):
        """Initialize ChestShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (x_train, y_train), (x_test, y_test) = self.download_data()
        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return ChestShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )
    
    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['224', '224', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['224', '224', '3']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Chest X-ray dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
    
    
    def download_data(self):
        """Download prepared dataset."""
        #current_dir = os.getcwd()
        current_dir = '~/'        
        
        tr_path = 'FL-SecFlow/envoy/chest-xray-pneumonia/chest_xray/train'
        val_path = 'FL-SecFlow/envoy/chest-xray-pneumonia/chest_xray/val'
        te_path = 'FL-SecFlow/envoy/chest-xray-pneumonia/chest_xray/test'
        
        train_path = os.path.join(current_dir, tr_path)
        valid_path = os.path.join(current_dir, val_path)
        test_path = os.path.join(current_dir, te_path)
        print(train_path)
        data_transforms = {
            'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4794, 0.4794, 0.4794), (0.2384, 0.2384, 0.2384)),
                    ]),
    
            'test': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4794, 0.4794, 0.4794), (0.2384, 0.2384, 0.2384)),
                    ]),
    
            'valid': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4794, 0.4794, 0.4794), (0.2384, 0.2384, 0.2384)),
                    ])
            }
        
        print('Chest X-Ray Pneumonia data loading...')

        train_dataset = datasets.ImageFolder((train_path), transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder((valid_path), transform=data_transforms['valid'])
        test_dataset = datasets.ImageFolder((test_path), transform=data_transforms['test'])

        val_test_data = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
        val_test_loader = torch.utils.data.DataLoader(val_test_data, batch_size=None, shuffle=False)

        x_train, y_train = zip(*[(x, y) for x, y in tqdm.tqdm(train_loader)])
        x_test, y_test = zip(*[(x, y) for x, y in tqdm.tqdm(val_test_loader)])
        
        print('Chest X-Ray Pneumonia data was loaded!')
        return (x_train, y_train), (x_test, y_test)