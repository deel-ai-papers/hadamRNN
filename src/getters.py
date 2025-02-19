import sys
from functools import reduce
from types import ModuleType
from typing import Any, Callable

from models import *
from extra_layers import *
from torch.nn import LSTM
from dataset_tools import copy_task, permuted_mnist, sequential_mnist, imdb, glue


def get_model(model_name):
    name = model_name.lower()
    models = {
        'binadamssm': BinadamSSM,
        'qrnn': QRNN,
        'qssm': QSSM,
        'qssmwithembeddings': QSSMwithEmbeddings,
        'binadamssmwithembeddings': BinadamSSMwithEmbeddings,
    }
    if name not in models:
        err_str = "{} is not a correct model name, accepted models are".format(model_name)
        for i, k in enumerate(models.keys()):
            if i == len(models) - 1 and i > 0: err_str += " and {}".format(k)
            else: err_str += " {},".format(k)
        err_str += " (case does not matter)"
        raise KeyError(err_str)
    return models[name]


def get_dataset(dataset_name):
    name = dataset_name.lower()
    datasets = {
        'addsequence': add_task.AddSequence,
        'copymemory': copy_task.CopyMemory,
        'pmnist': permuted_mnist.pMNIST,
        'penntreebank': penn_tree_bank.PennTreeBank,
        'smnist': sequential_mnist.sMNIST,
        'imdb': imdb.IMDB,
        'glue': glue.Glue,
    }
    if name not in datasets:
        err_str = "{} is not a correct dataset name, accepted datasets are".format(dataset_name)
        for i, k in enumerate(datasets.keys()):
            if i == len(datasets) - 1 and i > 0: err_str += " and {}".format(k)
            else: err_str += " {},".format(k)
        err_str += " (case does not matter)"
        raise KeyError(err_str)
    return datasets[name]


def get_object_from_modules(
    function_name: Any, sys_modules: ModuleType = sys.modules["__main__"]
) -> Any:
    """
    Given a string and a module, recovers the function
    from the modules if it's defined in the modules and the string
    starts with $.
    $ is the signifier for function name, strings not
    starting with $ are interpreted as normal strings not function
    names.
    function names with . are recursively iterated and imported. e.g.
    tensorflow.keras.losses.SparseCategoricalCrossentropy will:
    - get tensorflow from sys_modules
    - get keras from tensorflow
    - get losses from keras
    - get SparseCategoricalCrossentropy from losses

    Args:
        function_name (Any): name of function to be found in sys_modules. Strings that don't start with $ or non-strings are ignored
        and returned as is
        sys_modules: Module in which to search function name. Defaults to entry point

    Returns:
        Any: if function name is a string starting with $: the module of sys_modules named function_name
               Else: function_name
    """
    if isinstance(function_name, str) and function_name.startswith("$"):
        get_class = lambda name: reduce(
            getattr, name.split(".")[1:], __import__(name.partition(".")[0])
        )

        if "." in function_name:
            return get_class(function_name[1:])
        else:
            return getattr(sys_modules, function_name[1:])
    return function_name