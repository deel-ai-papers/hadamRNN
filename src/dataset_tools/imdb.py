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
import pickle as pkl
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from .dataset import Dataset


class MappedTensorDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor, transform):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx) 
        if self.transform is None:
            return data
        else:
            return self.transform(data)

class IMDB(Dataset):

    data_path = "downloaded_dataset/imdb"

    def __init__(self, seq_length=500, **kwargs):
        # vocabulary size
        assert (seq_length <= 500), ("IMDB maximum sentence length is 500, sorry 'bout that")
        train_size = 25000
        val_size = 12500
        test_size = 12500
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)
        self.embedding = True
        self._input_dimension = 10000
        self.word_embedding = torch.nn.Embedding(10000, 512, padding_idx=0)
        self._output_dimension = 1

    @property
    def input_dimension(self):
        return (self._input_dimension,)
    
    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def input_flat_dimension(self):
        return self._input_dimension

    @property
    def image_size(self):
        return None

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return self._output_dimension

    @num_outputs.setter
    def num_outputs(self, value):
        self._output_dimension = value

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

    def map_fn(self):
        def map_fn(data):
            x,y = data
            x = self.word_embedding(x)
            return (x,y)
        return map_fn
    
    def import_dataset(self):
        with open(data_path + '/imdb_data.pkl', 'rb') as f:
            dic = pkl.load(f)
        x_train = torch.from_numpy(dic['x_train'].astype(np.int32))[:,(-self.seq_length):]
        x_test = torch.from_numpy(dic['x_test'].astype(np.int32))[:,(-self.seq_length):]
        # split the X test data in validation and tes
        x_val = x_test[-self.val_size:]
        x_test = x_test[:self.test_size]

        y_train = torch.from_numpy(dic['y_train'].astype(np.float32))
        y_test = torch.from_numpy(dic['y_test'].astype(np.float32))
        # split the Y test data in validation and tes
        y_val = y_test[-self.val_size:]
        y_test = y_test[:self.test_size]

        train_ds = MappedTensorDataset(x_train, y_train, transform=None)
        test_ds = MappedTensorDataset(x_test, y_test, transform=None)
        val_ds = MappedTensorDataset(x_val, y_val, transform=None)

        return train_ds, val_ds, test_ds





data_path = "downloaded_dataset/imdb"



def load_data():
    with open(data_path + '/imdb_data.pkl', 'rb') as f:
        dic = pkl.load(f)
    x_train = torch.from_numpy(dic['x_train'].astype(np.float32))
    x_test = torch.from_numpy(dic['x_test'].astype(np.float32))
    y_train = torch.from_numpy(dic['y_train'].astype(np.float32))
    y_test = torch.from_numpy(dic['y_test'].astype(np.float32))

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    return train_ds, test_ds




def main():
    load_data()


if __name__ == "__main__":
    main()