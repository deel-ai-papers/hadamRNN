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
from .dataset import Dataset
from torch.utils.data import TensorDataset


class CopyMemory(Dataset):
    """Class to generate the sequence copy task dataset with some properties"""
    def __init__(self, train_size, val_size, test_size, seq_length, sent_length, vocabulary, **kwargs):
        self.vocabulary = vocabulary
        self.sent_length = sent_length
        self.naive_baseline = sent_length * np.log(vocabulary - 2) / seq_length
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)

    @property
    def input_dimension(self):
        return (self.vocabulary,)
    
    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def input_flat_dimension(self):
        return self.vocabulary

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

    # Generates Synthetic Data
    def Generate_Data_copy_task(self, size, length, K):

        assert length > 2 * K, (
            "copy_memory sequence length issue: the whole sequence should be "
            "strictly longer than twice its relevant part (the tokens to memorize)"
        )

        dim = self.input_flat_dimension
        seq = torch.randint(1, dim-1, size=(size, K))
        L = length - 2 * K
        zeros1 = torch.zeros((size, L), dtype=torch.int32)
        zeros2 = torch.zeros((size, K-1), dtype=torch.int32)
        zeros3 = torch.zeros((size, K+L), dtype=torch.int32)
        marker = (dim - 1) * torch.ones((size, 1), dtype=torch.int32)

        #x = torch.nn.functional.one_hot(
        #    torch.concatenate((seq, zeros1, marker, zeros2), axis=1),
        #    num_classes=self.input_flat_dimension
        #).to(torch.float32)

        x = torch.cat((seq, zeros1, marker, zeros2), axis=1)
        
        #y = torch.nn.functional.one_hot(
        #    torch.concatenate((zeros3, seq), axis=1),
        #    num_classes=(self.input_flat_dimension-1)
        #).to(torch.float32)
        y = torch.cat((zeros3, seq), axis=1)

        return x, y

    def import_dataset(self):
        np.random.seed(42)

        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        x_train, y_train = self.Generate_Data_copy_task(
                                    self.train_size, self.seq_length, self.sent_length)
        x_val, y_val = self.Generate_Data_copy_task(
                                    self.val_size, self.seq_length, self.sent_length)
        x_test, y_test = self.Generate_Data_copy_task(
                                    self.test_size, self.seq_length, self.sent_length)

        #train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        #val_ds = torch.utils.data.TensorDataset(x_val, y_val)
        #test_ds = torch.utils.data.TensorDataset(x_test, y_test)
        
        train_ds = OneHotBatchDataset(self.input_flat_dimension, x_train, y_train)
        val_ds = OneHotBatchDataset(self.input_flat_dimension, x_val, y_val)
        test_ds = OneHotBatchDataset(self.input_flat_dimension, x_test, y_test)

        print("-" * 60 + f"{type(self).__name__} loaded" + "-" * 60)

        return train_ds, val_ds, test_ds




class OneHotBatchDataset(TensorDataset):

    def __init__(self, num_classes, *tensors: torch.Tensor) -> None:
        super().__init__(*tensors)
        self.num_classes = num_classes

    def __getitem__(self, index):
        batch_x, batch_y = super().__getitem__(index)
        # one hot encode the batched sequence to predict
        batch_x = torch.nn.functional.one_hot(batch_x, num_classes=self.num_classes).to(torch.float32)
        return (batch_x, batch_y)