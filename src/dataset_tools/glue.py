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
import csv
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
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

class Glue(Dataset):

    #data_path = "datasets/imdb"

    def __init__(self, task = "sst2", use_embeddings = False, **kwargs):
        
        # upload  tokenizer 
        model_checkpoint = "bert-base-uncased"  
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.embedding = None
        if use_embeddings:
            hf_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
            self.embedding = hf_model.bert.embeddings.word_embeddings
            self.embedding_dim = self.embedding.embedding_dim
        self.vocab_size = self.tokenizer.vocab_size
                
        # Charger les données GLUE
        self.dataset = load_dataset("glue", task)
        '''print(self.dataset.keys())
        dd = next(iter(self.dataset['train']))
        print(dd)
        print(dd['label'])
        indexes  = np.arange(len(self.dataset['test']))
        predictions = [str(int(dd['label']))]*len(self.dataset['test'])
        '''
        BENCHMARK_SUBMISSION_FILENAMES = {
            "cola": "CoLA.tsv",
            "sst2": "SST-2.tsv",
            "mrpc": "MRPC.tsv",
            "stsb": "STS-B.tsv",
            "mnli": "MNLI-m.tsv",
            "mnli_mismatched": "MNLI-mm.tsv",
            "qnli": "QNLI.tsv",
            "qqp": "QQP.tsv",
            "rte": "RTE.tsv",
            "wnli": "WNLI.tsv",
            "ax": "AX.tsv",
            "glue_diagnostics": "AX.tsv",
        }
        
        self.output_filepath = os.path.join("glue_results",BENCHMARK_SUBMISSION_FILENAMES[task.lower()])
        '''
        with open(output_filepath, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(("index", "prediction"))
            writer.writerows(zip(indexes, predictions))
        aaa'''
        task2maxlength = {'sst2': 128,'cola': 128,'qqp': 128}
        seq_length = task2maxlength[task]

        # vocabulary size
        train_size = len(self.dataset['train'])
        val_size = len(self.dataset['validation'])
        test_size = len(self.dataset['test'])
        print(f"train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
        
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)
        self._input_dimension = 1
        if use_embeddings:
            self._output_dimension = self.embedding_dim
        else:
            self._output_dimension = 1
        

    def preprocess_function(self,examples):
        if 'sentence' in examples.keys():
            dd  = self.tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=self.seq_length)
        if 'question1' in examples.keys():
            concat_sentences = [q1+self.tokenizer.sep_token+self.tokenizer.sep_token+q2 for q1,q2 in zip(examples['question1'],examples['question2'])]
            dd  = self.tokenizer(concat_sentences, truncation=True, padding="max_length", max_length=self.seq_length)
        return dd

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
            x = data['input_ids']
            y = data['label']
            return (x,y)
        return map_fn
    
    def import_dataset(self):
        encoded_dataset = self.dataset.map(self.preprocess_function, batched=True)
        x_train = torch.from_numpy(np.asarray(encoded_dataset['train']['input_ids']).astype(np.int32))
        x_test = torch.from_numpy(np.asarray(encoded_dataset['test']['input_ids']).astype(np.int32))
        # split the X test data in validation and tes
        x_val = torch.from_numpy(np.asarray(encoded_dataset['validation']['input_ids']).astype(np.int32))
        if self.embedding is not None:
            x_train = self.embedding(x_train).detach()
            x_test = self.embedding(x_test).detach()
            x_val = self.embedding(x_val).detach()
        label_train = np.asarray(encoded_dataset['train']['label'])
        y_train = torch.from_numpy(label_train.astype(np.float32))
        y_test = torch.from_numpy(np.asarray(encoded_dataset['test']['label']).astype(np.float32))
        # split the Y test data in validation and tes
        y_val = torch.from_numpy(np.asarray(encoded_dataset['validation']['label']).astype(np.float32))
        print("train ",y_train.sum()/self.tr_size,"val ",y_val.sum()/self.va_size,"test ",y_test.sum()/self.te_size)
        
        class_sample_count = np.array(
            [len(np.where(label_train == t)[0]) for t in np.unique(label_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in label_train])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        self.sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))        
        train_ds = MappedTensorDataset(x_train, y_train, transform=None)
        test_ds = MappedTensorDataset(x_test, y_test, transform=None)
        val_ds = MappedTensorDataset(x_val, y_val, transform=None)
        return train_ds, val_ds, test_ds








def load_data(task = "sst2"):
    
    def preprocess_function(examples):
        dd  = tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=128)
        return dd

    # upload  tokenizer 
    model_checkpoint = "bert-base-uncased"  
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
                
    
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    
    embedding = hf_model.bert.embeddings.word_embeddings
    # Charger les données GLUE
    dataset = load_dataset("glue", task)
        
    task2maxlength = {'sst2': 128}
    seq_length = task2maxlength[task]

    # vocabulary size
    train_size = len(dataset['train'])
    val_size = len(dataset['validation'])
    test_size = len(dataset['test'])
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    x_train = torch.from_numpy(np.asarray(encoded_dataset['train']['input_ids']).astype(np.int32))
    x_train = embedding(x_train)
    x_test = torch.from_numpy(np.asarray(encoded_dataset['test']['input_ids']).astype(np.int32))
    # split the X test data in validation and tes
    x_val = torch.from_numpy(np.asarray(encoded_dataset['validation']['input_ids']).astype(np.int32))

    y_train = torch.from_numpy(np.asarray(encoded_dataset['train']['label']).astype(np.float32))
    y_test = torch.from_numpy(np.asarray(encoded_dataset['test']['label']).astype(np.float32))
    # split the Y test data in validation and tes
    y_val = torch.from_numpy(np.asarray(encoded_dataset['validation']['label']).astype(np.float32))

    train_ds = MappedTensorDataset(x_train, y_train, transform=None)
    test_ds = MappedTensorDataset(x_test, y_test, transform=None)
    val_ds = MappedTensorDataset(x_val, y_val, transform=None)
    return dataset['train'], dataset['validation'], dataset['test'], seq_length, train_size, val_size, test_size


def main():
    load_data()


if __name__ == "__main__":
    main()