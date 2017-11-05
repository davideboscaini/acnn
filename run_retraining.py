# modules
import os
import cPickle
import numpy as np
import lasagne.layers as LL

# personal modules
from datasets import dataset_class
import models
import trainings

# ------------------------------------------------------------------------------------------------- editable parameters
start_from_epoch = 25
path_pkls = 'path_to_pkls'
# ---------------------------------------------------------------------------------------------------------------------

# path to the pickles
name_pkl = os.path.join(path_pkls, 'pkls', 'epoch=%04d.pkl' % start_from_epoch)

# load the pickle containing the learned parameters and other statistics
with open(name_pkl, 'rb') as f:
    [paths, param_arch, param_cost, param_updates, param_train,
     cost_train_avg, acc_train_avg, cost_test_avg, acc_test_avg,
     keys_net, values_net, keys_updates, values_updates] = cPickle.load(f)

# quick sanity check
print(keys_net)
print(keys_updates)

# update the paths
paths['pkls'] = os.path.join(paths['exp'], 'pkls_from_epoch=%04d' % start_from_epoch)
paths['preds'] = os.path.join(paths['exp'], 'preds_from_epoch=%04d' % start_from_epoch)

# update the training parameters
param_train['start_from_epoch'] = start_from_epoch

# set the random seed
np.random.seed(42)

# load the dataset
ds = dataset_class(os.path.join(paths['exp'], 'files_train.txt'),
                   os.path.join(paths['exp'], 'files_test.txt'),
                   paths['descs'],
                   paths['patch_ops'],
                   paths['labels'],
                   param_train['epoch_size'])

# compute the model
funcs, cla, updates = models.model_class(ds, paths, param_arch, param_cost, param_updates, param_train)

# restore network values
LL.set_all_param_values(cla, values_net, trainable=False)

# restore updates values
for k, v in zip(updates.keys(), values_updates):
    k.set_value(v)

# re-training
cost_train_avg, acc_train_avg, cost_test_avg, acc_test_avg = \
    trainings.train_class(ds, paths, funcs, cla, updates, param_arch, param_cost, param_updates, param_train)
