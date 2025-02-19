import os.path as path
import yaml

from torch.optim import Optimizer

from getters import get_object_from_modules


class Config:

    def __init__(self, conf_dic=None, conf_file=None, instantiate=True):
        if conf_dic is None and conf_file is None: return 
        if conf_dic is not None:
            self.config = conf_dic
        elif conf_file is not None:
            self.config = self.make_config_out_of_file(conf_file)
        for k, dic in self.config.items():
            if k.lower() == 'dataset': self.dataset = self.format_dic(dic)
            if k.lower() == 'model': self.model = self.format_dic(dic)
            if k.lower() == 'train': self.train = self.format_dic(dic)
            if k.lower() == 'project': self.project = dic
        if instantiate:
            self.instantiate_activation()
            self.instantiate_loss()
            self.instantiate_metrics()

    def make_config_out_of_file(self, conf_file):
        ext = path.splitext(conf_file)[-1]
        extensions = ['.yaml']
        if ext not in extensions:
            err_str = "{} is not a correct file extension, accepted files are".format(conf_file)
            for i, k in enumerate(extensions.keys()):
                if i == len(extensions) - 1 and i > 0: err_str += " and {}".format(k)
                else: err_str += " {},".format(k)
            raise KeyError(err_str)
        
        if ext == '.yaml': return self.make_config_out_of_yaml(conf_file)

    def make_config_out_of_yaml(self, conf_file):
        with open(conf_file, 'r') as f:
            dic = yaml.safe_load(f)
        return dic

    def format_dic(self, dic):
        d = {k.lower(): v for k, v in dic.items()}
        for k, v in d.items():
            if isinstance(v, str) and v.startswith('$'):
                d[k] = get_object_from_modules(v)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, str) and item.startswith('$'):
                        v[i] = get_object_from_modules(item)
            if isinstance(v, dict):
                for kk, vv, in v.items():
                    if isinstance(vv, str) and vv.startswith('$'):
                        v[kk] = get_object_from_modules(vv)
        return d
        
    def instantiate_optimizer(self, params):
        if not self.train['optimizer_config']: self.train['optimizer'] = self.train['optimizer'](params=params)
        else: self.train['optimizer'] = self.train['optimizer'](params=params, **self.train['optimizer_config'])
        
    def instantiate_scheduler(self):
        optimizer = self.train['optimizer']
        if not isinstance(optimizer, Optimizer):
            raise AttributeError("Error when trying to instantiate the torch scheduler ; "
                                 "optimizer has not been initialized yet")
        if not self.train['scheduler_config']: self.train['scheduler'] = self.train['scheduler'](optimizer=optimizer)
        else: self.train['scheduler'] = self.train['scheduler'](optimizer=optimizer, **self.train['scheduler_config'])

    def instantiate_loss(self):
        if not self.train['loss_config']: self.train['loss_fn'] = self.train['loss_fn']()
        else: self.train['loss_fn'] = self.train['loss_fn'](**self.train['loss_config'])

    def instantiate_activation(self):
        if 'activation' in self.model.keys() and self.model['activation'] is not None:
            if ('activation_config' in self.model.keys()) and (self.model['activation_config']):
                self.model['activation'] = self.model['activation'](**self.model['activation_config'])
            else:
                self.model['activation'] = self.model['activation']()
        if 'activation_final' in self.model.keys() and self.model['activation_final'] is not None:
            if ('activation_final_config' in self.model.keys()) and (self.model['activation_final_config']):
                self.model['activation_final'] = self.model['activation_final'](**self.model['activation_final_config'])
            else:
                self.model['activation_final'] = self.model['activation_final']()

    def instantiate_metrics(self):
        if self.train['metrics'] is not None:
            for k, v in self.train['metrics'].items():
                if self.train['metrics_config'] is not None:
                    self.train['metrics'][k] = v(**self.train['metrics_config'])
                else: self.train['metrics'][k] = v()

    def get_device(self):
        try:
            return self.train['torch_device']
        except AttributeError:
            raise
        except KeyError:
            return 'cpu'