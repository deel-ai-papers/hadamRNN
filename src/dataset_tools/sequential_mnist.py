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
import torch
import numpy as np
from torchvision import datasets, transforms
from .dataset import Dataset


class sMNIST(Dataset):
    """Class to generate the MNIST dataset with some properties"""

    def __init__(self, **kwargs):
        seq_length = self.image_size * self.channels
        train_size = 50000
        val_size = 10000
        test_size = 10000
        if 'train_size' in kwargs.keys():
            if 'val_size' in kwargs.keys():
                assert kwargs['train_size'] + kwargs['val_size'] == 60000, (
                    "validation and train sets should contain in a whole 60000 examples")
            else: kwargs['val_size'] = 60000 - kwargs['train_size']
            train_size = kwargs['train_size']
            val_size = kwargs['val_size']
            kwargs.pop('train_size')
            kwargs.pop('val_size')
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)

    @property
    def input_dimension(self):
        return (1,)

    @property
    def input_flat_dimension(self):
        return 1

    @property
    def image_size(self):
        return 28 * 28

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return 10

    @property
    def test_size(self):
        return self.te_size

    @property
    def train_size(self):
        return self.tr_size

    @property
    def val_size(self):
        return self.va_size

    def get_train_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.test_ds

    def get_val_ds(self):
        return self.val_ds

    def import_dataset(self):

        # this transform allows to download the mnist images in the flattened shape
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ])
        
        print("-" * 43 + f" Loading {type(self).__name__} " + "-" * 43)

        train_ds, val_ds = torch.utils.data.random_split(
            datasets.MNIST("downloaded_dataset/mnist", train=True, download=True, transform=transform),
            [self.train_size, self.val_size]
        )
        test_ds = datasets.MNIST("downloaded_dataset/mnist", train=False, transform=transform)

        print("-" * 43 + f" {type(self).__name__} loaded " + "-" * 43)

        return train_ds, val_ds, test_ds
