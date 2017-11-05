# modules
import os
import numpy as np
# personal modules
import splittings
from datasets import dataset_class
import models
import trainings

# ------------------------------------------------------------------------------------------------- editable parameters
# path to the folder containing the code:
path_code = 'path_to_the_code'
# path to the folder containing the data:
path_data = 'path_to_the_data'
# experiment name:
name_exp = 'exp00'  # default: exp00
# architecture parameters:
param_arch = dict()
param_arch['type'] = 1  # possible choices: 0, 1, or 2
param_arch['flag_batchnorm'] = True
param_arch['non_linearity'] = 'log_softmax'  # possible choices: 'softmax' or 'log_softmax'
# cost function parameters:
param_cost = dict()
param_cost['cost_func'] = 'cross_entropy'
param_cost['mu'] = 1e-05
# updates parameters:
param_updates = dict()
param_updates['method'] = 'adam'  # possible choices: 'momentum', 'adagrad', 'adadelta', or 'adam'
param_updates['learning_rate'] = 1e-03
param_updates['momentum'] = None  # required only if methods_updates = 'momentum'
param_updates['beta1'] = 0.9
param_updates['beta2'] = 0.999
param_updates['epsilon'] = 1e-08
# training parameters:
param_train = dict()
param_train['epoch_size'] = 80  # default: 80
param_train['n_epochs'] = 1000
param_train['freq_viz_train'] = 1
param_train['freq_viz_test'] = 10
param_train['flag_save_pkls'] = True
param_train['freq_save_pkls'] = 10
param_train['flag_save_preds'] = True
param_train['freq_save_preds'] = 10
# ---------------------------------------------------------------------------------------------------------------------

# input paths (where to load data from):
paths = dict()
# path to the input descriptors
paths['descs'] = os.path.join(path_data, 'datasets/FAUST_registrations/data_used_for_paper/diam=200/descs/shot')
# path to the ACNN patch operators
paths['patch_ops'] = os.path.join(path_data, 'datasets/FAUST_registrations/data_used_for_paper/diam=200/patch_ops_aniso',
                                  'alpha=100_nangles=016_ntvals=005_tmin=6.000_tmax=24.000_thresh=99.900_norm=L1')
# path to the labels
paths['labels'] = os.path.join(path_data, 'datasets/FAUST_registrations/data_used_for_paper/diam=200/labels')

# output paths (where to save the results):
# path where to save the experiment information
paths['exp'] = os.path.join(path_code, 'experiments', name_exp)
paths['pkls'] = os.path.join(paths['exp'], 'pkls')
paths['preds'] = os.path.join(paths['exp'], 'preds')

# splits training and testing sets
splittings.default_split_FAUST(paths['exp'])

# set the random seed
np.random.seed(42)

# computes the dataset (loads descriptors, patch operators and labels)
print('\n[i] computing the dataset:')
ds = dataset_class(os.path.join(paths['exp'], 'files_train.txt'),
                   os.path.join(paths['exp'], 'files_test.txt'),
                   paths['descs'],
                   paths['patch_ops'],
                   paths['labels'],
                   param_train['epoch_size'])

# quick sanity check
print('\n[i] quick sanity check:')
for item in ds.train_iter():
    print '[i] desc:   min = %e, max = %e' % (item[0].min(), item[0].max())
    print '[i] patch:  min = %e, max = %e' % (item[1].min(), item[1].max())
    print '[i] labels: min = %e, max = %e' % (item[2].min(), item[2].max())
    break

# computes the model
print('\n[i] computing the model:')
funcs, cla, updates = models.model_class(ds, paths, param_arch, param_cost, param_updates, param_train)

# starts training
print('\n[i] starting training:')
cost_train_avg, acc_train_avg, cost_test_avg, acc_test_avg = \
    trainings.train_class(ds, paths, funcs, cla, updates, param_arch, param_cost, param_updates, param_train)
