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
from abc import ABC, abstractmethod



class Dataset(ABC):
    def __init__(self, train_size, val_size, test_size, seq_length, **kwargs):
        """
        This is an abstract data-set class that standardizes the interfaces of datasets within this folder.
        Like any other abstract classes, you should not need to instantiate this class directly, just its children
        """
        self.seq_length = seq_length
        self.tr_size = train_size
        self.va_size = val_size
        self.te_size = test_size
        self.train_ds, self.val_ds, self.test_ds = self.import_dataset()

    def to(self, device, train=False, val=False, test=False):
        if train: self.train_ds = self.train_ds.to(device)
        if val: self.val_ds = self.val_ds.to(device)
        if test: self.test_ds = self.test_ds.to(device)


    @property
    @abstractmethod
    def input_dimension(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def input_flat_dimension(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def image_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def channels(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_outputs(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def test_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def val_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def train_size(self):
        raise NotImplementedError

    @abstractmethod
    def get_val_ds(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_ds(self):
        raise NotImplementedError

    @abstractmethod
    def get_test_ds(self):
        raise NotImplementedError

    @abstractmethod
    def import_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return 