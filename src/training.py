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
try :
    import wandb
except:
    wandb = None
import csv
import numpy as np

from tqdm import tqdm
from torchmetrics import Accuracy

#from models import HadamRNN
#from layers import OrthogonalRNN, ortho_penalty
from quantized_layers import quantize
from utils import is_quantized, make_pow2_hadamard_matrix



def train(model, dataset, n_epochs, batch_size, loss_fn, optimizer, metrics=None, lambda_orth=0., scheduler=None, get_gradients=False, track_flips=False, display_every=None, display_epoch=False, use_wandb=False, torch_device=None, use_tqdm=True, compute_sv=True, alternate_gradients=False, **kwargs):
    save_test_output= False
    if alternate_gradients: model.initialize_gradients()
    sampler = dataset.sampler if hasattr(dataset, 'sampler') else None
    shuffle = True if sampler is None else False
    train_loader = torch.utils.data.DataLoader(
        dataset.train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler
    )

    validation = hasattr(dataset, "val_ds") and (dataset.val_ds is not None) and (len(dataset.val_ds) > 0)
    ds_size = len(dataset.train_ds)
    n_batches = ds_size // batch_size

    best_val_loss = 1e7
    best_accuracy = 0.
    
    for epoch in range(n_epochs):
        if model.__class__.__name__ == 'BinadamRNN':
            assert torch.unique(model.recurrent_layer.weight).shape[0] <= 3, "model error"
        model.train()

        # set the keys for training data we want to register, averaged over batches
        stat_epoch = {"loss": 0., "only_loss": 0.}
        if metrics:
            stat_epoch.update({name: 0. for name in metrics.keys()})
        if get_gradients:
            stat_epoch.update({'gradients/recurrent_1_'+name: 0. for name in model.recurrent_layer._parameters.keys()})
            if model.__class__.__name__ == 'BinadamRNN':
                stat_epoch.update({'gradients/recurrent_2_'+name: 0. for name in model.recurrent_layer_2._parameters.keys()})
                stat_epoch.update({'gradients/input_2_'+name: 0. for name in model.input_layer_2._parameters.keys()})
        if track_flips:
            stat_epoch['flips'] = 0

        if use_tqdm: train_loader = tqdm(train_loader)

        # enter the loop over batches
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # set data to be displayed next to the progress bar
            if use_tqdm:
                train_loader.set_description(f"Epoch {epoch+1}/{n_epochs}")
                tqdm_postfix = {"avg loss": stat_epoch["loss"], "lr": optimizer.param_groups[0]['lr']}
                if metrics and 'accuracy' in metrics.keys():
                    tqdm_postfix["avg accuracy"] = stat_epoch["accuracy"]
                train_loader.set_postfix(tqdm_postfix)

            # make a training step, and record additional data if required
            stat_batch = training_step(batch_x, batch_y, model, optimizer, loss_fn, metrics=metrics, lambda_orth=lambda_orth, get_gradients=get_gradients, track_flips=track_flips, torch_device=torch_device)

            # update training data: average loss, metrics etc
            stat_epoch = batch_update(stat_epoch, stat_batch, batch_idx)

            # display additional data
            if display_every and (batch_idx % display_every == 0):
                display_train_data(loss=stat_batch['loss'], only_loss=stat_batch['only_loss'], batch_idx=batch_idx, n_batches=n_batches, epoch=epoch)

        if scheduler is not None:
            scheduler.step()

        if compute_sv:
            sv_vals = compute_svd(model)
            sv_max = sv_vals.max()
            sv_min = sv_vals.min()
            sv_ratio = sv_min / sv_max

        if validation:
            val_batch_size = len(dataset.val_ds) // 10
            stat_val = evaluate(dataset.val_ds, val_batch_size, model, loss_fn, metrics=metrics, kind='validation', torch_device=torch_device)
            print(stat_val)
            has_to_save = False
            if stat_val['val_loss'] < best_val_loss:
                best_val_loss = stat_val['val_loss']
                has_to_save = True
                postfix_save = f".best_loss_{best_val_loss:.3f}_epoch_{epoch}.tsv"
            if ('val_accuracy' in  stat_val.keys()) and (stat_val['val_accuracy'] > best_accuracy):
                best_accuracy = stat_val['val_accuracy']
                postfix_save = f".best_acc_{best_accuracy:.2f}_epoch_{epoch}.tsv"
                has_to_save = True
            if save_test_output and has_to_save:
                    print(dataset.te_size)
                    outs = predict(dataset.test_ds, val_batch_size, model, loss_fn, kind='test', torch_device=torch_device)
                    indexes  = np.arange(outs.shape[0])
                    predictions = ['0' if oo < 0. else '1' for oo in outs]
                    output_filepath = dataset.output_filepath + postfix_save #"SST-2.tsv"
                    with open(output_filepath, "w") as f:
                        writer = csv.writer(f, delimiter="\t")
                        writer.writerow(("index", "prediction"))
                        writer.writerows(zip(indexes, predictions))
                    print("TSV saved for val loss: ", best_val_loss, "acc loss: ", best_accuracy, " at epoch: ", epoch)
            
        if display_epoch:
            display_train_data(epoch_loss=stat_epoch['loss'], epoch_only_loss=stat_epoch['only_loss'], **stat_val)

        if alternate_gradients:
            try:
                model.alternate_gradients()
            except AttributeError:
                print("Warning: model {} has no method alternate_gradients. Please disable "
                      "alternate_gradients option or write a custom alternate_gradients method for your model")

        if use_wandb and (wandb is not None):
            # TODO mettre une option pour customiser les données qu'on veut envoyer sur wandb
            wandb_dic = stat_epoch.copy()
            wandb_dic["lr"] = optimizer.param_groups[0]['lr']
            if validation:
                wandb_dic.update(stat_val)
            if hasattr(model.recurrent_layer, "angle"):
                try:
                    wandb_dic["modele angle"] = model.recurrent_layer.angle.item()
                except:
                    wandb_dic["modele average angle"] = model.recurrent_layer.angle.mean()
                wandb_dic["modele sign vector average"] = model.recurrent_layer.sign_vector.abs().mean()
            if hasattr(dataset, "naive_baseline"):
                wandb_dic["CCE baseline"] = dataset.naive_baseline
            if compute_sv:
                wandb_dic["eigenvalues/sv_min"] = sv_min
                wandb_dic["eigenvalues/sv_max"] = sv_max
                wandb_dic["eigenvalues/sv_ratio"] = sv_ratio
            wandb.log(wandb_dic)

    return model


def training_step(batch_x, batch_y, model, optimizer, loss_fn, metrics, lambda_orth=0., get_gradients=False, track_flips=False, torch_device=None, **kwargs):

    model.train()
    batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device).view(-1)
    predictions = model(batch_x).view(-1, model.output_layer.out_features).squeeze()
    loss = loss_fn(predictions, batch_y)
    if not hasattr(model.recurrent_layer, "sign_vector"): track_flips = False

    only_loss = loss.item()

    # regularization
    if lambda_orth > 0.:
        regul_loss = ortho_penalty(
            quantize(model.recurrent_layer.weight, num_bits=model.recurrent_layer.num_bits))
        loss += lambda_orth * regul_loss

    # update weights
    optimizer.zero_grad()
    if track_flips: sgns = torch.sign(model.recurrent_layer.sign_vector)
    loss.backward()
    optimizer.step()

    #model.apply_weight_constraints_()

    stat_batch = {"loss": loss.item(), "only_loss": only_loss}

    if metrics:
        metrics_values = compute_metrics(metrics, predictions, batch_y, torch_device=torch_device)
        stat_batch.update(metrics_values)

    if get_gradients:
        grads = get_layer_gradients(model.recurrent_layer, layer_name="recurrent_1")
        stat_batch.update(grads)
        if model.__class__.__name__ == "BinadamRNN":
            grads = get_layer_gradients(model.recurrent_layer_2, layer_name="recurrent_2")
            stat_batch.update(grads)
            grads = get_layer_gradients(model.input_layer_2, layer_name="input_2")
            stat_batch.update(grads)
            

    if track_flips:
        n_flips = torch.sum(torch.abs(torch.sign(model.recurrent_layer.sign_vector) - sgns) / 2.)
        stat_batch['flips'] = n_flips.item()

    return stat_batch


def batch_update(stat_epoch, stat_batch, batch_idx):
    for k in stat_batch.keys():
        if k not in stat_epoch.keys(): raise KeyError("key {} is not an epoch statistic".format(k))
        stat_epoch[k] = (batch_idx * stat_epoch[k] + stat_batch[k]) / (batch_idx + 1)
    return stat_epoch


def evaluate(dataset, batch_size, model, loss_fn, metrics=None, kind='validation', torch_device=None, **kwargs):
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=True
    )

    if kind == 'validation': prefix = 'val_'
    elif kind == 'test': prefix = 'test_'
    else: raise AttributeError("evaluation kind {} unknown".format(kind))

    model.eval()
    running_vloss = 0.
    if metrics: metric_values = {(prefix+name): 0. for name in metrics.keys()}

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(torch_device), vlabels.to(torch_device).view(-1)
            voutputs = model(vinputs).view(-1, model.output_layer.out_features).squeeze()
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            if metrics:
                metric_batch = compute_metrics(metrics, voutputs, vlabels, torch_device=torch_device)
                for k in metric_batch.keys():
                    metric_values[prefix+k] += metric_batch[k]
                
    val_loss = running_vloss / (i + 1)
    stat_eval = {prefix+"loss": val_loss.item()}
    if metrics: stat_eval.update({k:(v / (i+1)) for k, v in metric_values.items()})
    return stat_eval

def predict(dataset, batch_size, model, loss_fn, metrics=None, kind='validation', torch_device=None, **kwargs):
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=False
    )

    if kind == 'validation': prefix = 'val_'
    elif kind == 'test': prefix = 'test_'
    else: raise AttributeError("evaluation kind {} unknown".format(kind))

    model.eval()
    vouts = []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(torch_device), vlabels.to(torch_device).view(-1)
            voutputs = model(vinputs).view(-1, model.output_layer.out_features).squeeze()
            vouts.append(voutputs.detach().cpu().numpy())   
    return np.concatenate(vouts,axis=0)


def compute_metrics(metrics, batch_preds, batch_y, torch_device=None):
    """ metrics should be a dictionnary of the form {name: metric}, with name being a string
        and metric a torchmetrics.metric instance
    """

    metrics_values = {}

    for name, metric in metrics.items():
        metric.to(torch_device)
        batch_preds, batch_y = batch_preds.to(torch_device), batch_y.to(torch_device)
        metrics_values[name] = metric(batch_preds, batch_y).item()

    return metrics_values


def get_layer_gradients(layer, layer_name='', operation='average_norm'):
    gradients = {}
    for name, p in layer._parameters.items():
        if p is None: continue
        grad = p.grad
        if grad is None: grad = torch.tensor(0., dtype=p.dtype)
        if operation == 'average_norm':
            gradients['gradients/' + layer_name + '_' +name] = torch.mean(torch.abs(grad))
        else: raise ValueError("operation {} is unknown. Allowed operations are \"average_norm\"".format(operation))
        # TODO autres opérations éventuelles
    return gradients


def display_train_data(round=3, **stats):
    display_str = ""
    for k, v in stats.items():
        display_str += "\t  {}: {}".format(k, np.round(v, round))
    print(display_str)


def compute_svd(model):
    weights = model.recurrent_layer.weight
    if model.__class__.__name__ == 'proj_net' and model.recurrent_layer.__class__.__name__ == 'QLinear':
        weights = quantize(weights)
    if model.recurrent_layer.__class__.__name__ == 'Binadamard' or model.recurrent_layer.__class__.__name__ == 'ParametrizedBinadamard':
        weights = model.recurrent_layer.scale * weights
    return torch.linalg.svdvals(weights)