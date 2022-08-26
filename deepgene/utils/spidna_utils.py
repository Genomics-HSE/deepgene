# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import builtins
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import gc
import logging


def convert_haplo_to_geno(snp):
    return np.vstack([np.sum(snp[n:n + 2], 0) for n in range(0, snp.shape[0], 2)])


def debug_gpu(msg=''):
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    logging.debug(f'{msg} {type(obj)} {obj.size()}')
            except:
                pass


def remove_maf_folded(snp, pos, maf, relative_position=True):
    if maf != 0:
        num_sample = snp.shape[0]
        if (snp == 2).any():
            num_sample *= 2
        if relative_position:
            abs_pos = np.cumsum(pos)
        row_sum = np.sum(snp, axis=0)
        a = np.logical_and(row_sum != 0, row_sum != num_sample)
        b = row_sum > num_sample * maf
        c = num_sample - row_sum > num_sample * maf
        snp = snp[:, np.logical_and(np.logical_and(a, b), c)]
        if pos is None:
            return snp, pos
        
        if relative_position:
            abs_pos = abs_pos[np.logical_and(np.logical_and(a, b), c)]
            pos = abs_pos - np.concatenate(([0], abs_pos[:-1]))
        else:
            pos = pos[np.logical_and(np.logical_and(a, b), c)]
    return snp, pos


def remove_maf(snp, pos, maf, relative_position=True):
    if maf != 0:
        if relative_position:
            pos = np.cumsum(pos)
        row_sum = np.sum(snp, axis=0)
        a = np.logical_and(row_sum != 0, row_sum != snp.shape[0])
        b = row_sum > snp.shape[0] * maf
        snp = snp[:, np.logical_and(a, b)]
        if pos is None:
            return snp, pos
        pos = pos[np.logical_and(a, b)]
        if relative_position:
            pos = pos - np.concatenate(([0], pos[:-1]))
    return snp, pos


def transform_to_min_major(data):
    row_sum = np.sum(data, axis=0)
    wrong_encoding = row_sum > data.shape[0] / 2
    data[:, wrong_encoding] = 1 - data[:, wrong_encoding]
    return data


def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        # nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.ModuleList):
        for sub_module in module:
            init_weights(sub_module)


def print(*args):
    """Overwrite the print function so flush is always true.

    It prevents print function from using the buffer and prevent buffer issues of slurm.
    """
    builtins.print(*args, flush=True)


def count_parameters(model):
    """Count the number of learnable parameter in an object of the :class:torch.nn.Module: class.

    Arguments:
        model (nn.Module):
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dict_from_json(filepath):
    """Load a json in a dictionnary.

    The dictionnary contains the overall parameters used for the simulation (e.g. path to the data folder, number of epoch). The json can be created from a dictionnary using ``save_dict_in_json``.

    Arguments:
        filepath (string): filepath to the json file
    """
    return json.loads(open(filepath, 'r').read())


def save_dict_in_json(filepath, params):
    """Save a dictionnary into a json file.

    Arguments:
        filepath (string): filepath of the json file
        params(dict): dictionnary containing the overall parameters used for the simulation (e.g. path to the data folder, number of epoch...)
    """
    with open(filepath, 'w') as file_handler:
        json.dump(params,
                  file_handler,
                  indent=4)  # for pretty printing.


def standardize(parameters, mean, std):
    """Standardize the demographic parameters following :math:`\theta = \frac{\theta - \mu}{\sigma}.`

    Means and standard deviation of each parameters should be precomputed over the training set.

    Arguments:
        parameters (torch.Tensor or numpy.ndarray): :math:`n \times m` parameters to standardize
        mean (torch.Tensor or numpy.ndarray): mean of each parameter computed over the training in a vector of size :math:`m`
        std (torch.Tensor or numpy.ndarray): standard deviation of each parameter computed over the training in a vector of size :math:`m`
    """
    return ((parameters - mean) / std)


def cut_and_cat(tensor1, tensor2):
    """
    To describe.
    """
    if tensor1.size() == torch.Tensor([]).size():
        return tensor2
    elif tensor2.size() == torch.Tensor([]).size():
        return tensor1
    dif = tensor1.size(2) - tensor2.size(2)
    
    if dif > 0:
        return torch.cat((tensor1[:, :, dif // 2:-dif // 2], tensor2), 1)
    elif dif < 0:
        return torch.cat((tensor1, tensor2[:, :, -dif // 2:dif // 2]), 1)
    else:
        return torch.cat((tensor1, tensor2), 1)


def generate_filename(index, model_name, num_scenario=None, num_replicates=None,
                      return_abs_path=False, format_scen=None, format_rep=None):
    """Given parameters of a simulation, returns the name of the npz file.
        If return_abs_path is True, it returns the full path

    Parameters
    ----------
    index : int, or tuple
        i-th simulation. If tuple: (scen_id, rep_id).
    model_name : str
        Name of the population genetic model.
    num_scenario : int
        Number of scenario in the given model.
    num_replicates : int
        Number of replicates per scenario.
    return_abs_path : bool
        Whether to return the full path from the model directory (True) or just the filename (False).
    format_scen: str
        If not None, pass a string to set the format of the scenario identifier in the filename.
        e.g. "0>3"
    format_rep: str
        If not None, pass a string to set the format of the replicate identifier in the filename.
        e.g. "0>2"

    Returns
    -------
    str
        Path to or only filename of the npz file.

    """
    if isinstance(index, tuple):
        scen_id, rep_id = index
    else:
        scen_id = index // num_replicates
        rep_id = index % num_replicates
    if format_scen is None and num_scenario is not None:
        format_scen = "0>" + str(np.ceil(np.log10(num_scenario)).astype(int))
    else:
        format_scen = "0>"
    if format_rep is None and num_replicates is not None:
        format_rep = "0>" + str(np.ceil(np.log10(num_replicates)).astype(int))
    else:
        format_rep = "0>"
    scenario_dir = f'scenario_{scen_id:{format_scen}}'
    npz_file = f'{model_name}_{scen_id:{format_scen}}_{rep_id:{format_rep}}.npz'
    if return_abs_path:
        return os.path.join(model_name,
                            f"scenario_{scen_id:{format_scen}}",
                            npz_file)
    else:
        return npz_file


def rescale(tensors, means, stds, param_2_lean, param_2_log, label=""):
    """Rescale a tensor of shape (n x p) by reversing the lean centering and standardization.
    returns tensor * stds + means

    Parameters
    ----------
    tensors : torch.Tensor
        predicted or target values.
    means : pd.Series
        Series with the mean for the different learned parameters.
    stds : pd.Series
        Series with the standard deviation for the different learned parameters.
    param_2_lean : list
        List of learned parameters of size (p,).
    param_2_log : list
        List of paramaters to get the exponential from.
    label : str
        String to append to the name of the columns which will be the name of the parameters.

    Returns
    -------
    pd.DataFrame
        Dataframe with the parameters in columns.

    """
    
    tmp = pd.DataFrame(tensors.detach().cpu().numpy(), columns=param_2_lean)
    tmp_rescale = tmp * stds + means
    for pname in param_2_lean:
        if pname in param_2_log:
            tmp_rescale[pname] = np.exp(tmp_rescale[pname])
    if label != "":
        tmp_rescale.columns = ["_".join([i, label]) for i in tmp_rescale.columns]
    return tmp_rescale
