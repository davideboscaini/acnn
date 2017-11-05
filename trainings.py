# system modules
import os
import time
import cPickle
import logging

# numerical modules
import numpy as np
import scipy.io

# Lasagne related modules
import lasagne.layers as LL


def train_class(ds, paths, funcs, cla, updates, param_arch, param_cost, param_updates, param_train):

    # creates a log file containing the training behaviour,
    # saves it to file
    formatter = logging.Formatter('%(asctime)s %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('log_training')
    if 'start_from_epoch' in param_train:
        name_tmp = 'training_from_epoch=%04d.log' % (param_train['start_from_epoch'])
    else:
        name_tmp = 'training.log'
    path_tmp = os.path.join(paths['exp'], name_tmp)
    if not os.path.isfile(path_tmp):
        file_handler = logging.FileHandler(path_tmp, mode='w')
    else:
        raise Exception('[e] the log file file ', name_tmp, ' already exists!')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # and shows it to screen
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.info("Training stats:")
    
    cost_train_avg = []
    cla_train_avg = []
    reg_train_avg = []
    grad_norm_train_avg = []
    acc_train_avg = []
    cost_test_avg = []
    grad_norm_test_avg = []
    acc_test_avg = []

    for i_ in xrange(param_train['n_epochs']):

        if 'start_from_epoch' in param_train:
            i = i_ + param_train['start_from_epoch']
        else:
            i = i_
        
        cost_train = []
        cla_train = []
        reg_train = []
        cost_test = []
        grad_norm_train = []
        grad_norm_test = []
        acc_train = []
        acc_test = []

        tic = time.time()
        
        for x in ds.train_iter():
            tmp = funcs['train'](*x)
            cost_train.append(tmp[0])
            cla_train.append(tmp[1])
            reg_train.append(tmp[2])
            grad_norm_train.append(tmp[3])
            acc_train.append(tmp[4])

        if ((i + 1) % param_train['freq_viz_train']) == 0:
            cost_train_avg.append(np.mean(cost_train))
            cla_train_avg.append(np.mean(cla_train))
            reg_train_avg.append(np.mean(reg_train))
            grad_norm_train_avg.append(np.mean(grad_norm_train))
            acc_train_avg.append(np.mean(acc_train))
            string = "[TRN] epoch = %03i, cost = %3.2e (cla = %3.2e, reg = %3.2e), |grad| = %.2e, acc = %02.2f%% (%03.2fs)" % \
                (i + 1, cost_train_avg[-1], cla_train_avg[-1], reg_train_avg[-1], grad_norm_train_avg[-1], 100.*acc_train_avg[-1], time.time() - tic)
            logger.info(string)
            
        if ((i + 1) % param_train['freq_viz_test']) == 0:

            tic = time.time()

            for x in ds.test_fwd():
                tmp = funcs['fwd'](*x)
                cost_test.append(tmp[0])
                grad_norm_test.append(tmp[1])
                acc_test.append(tmp[2])
          
            cost_test_avg.append(np.mean(cost_test))
            grad_norm_test_avg.append(np.mean(grad_norm_test))
            acc_test_avg.append(np.mean(acc_test))
            string = "[TST] epoch = %03i, cost = %3.2e, |grad| = %.2e, acc = %02.2f%% (%03.2fs)" % \
                (i + 1, cost_test_avg[-1],  grad_norm_test_avg[-1],  100.*acc_test_avg[-1],  time.time() - tic)
            logger.info(string)
            
        if param_train['flag_save_pkls']:
            if ((i + 1) % param_train['freq_save_pkls']) == 0:
                if not os.path.isdir(paths['pkls']):
                    os.makedirs(paths['pkls'])
                name_dump = "%s/epoch=%04d.pkl" % (paths['pkls'], i + 1)
                keys_net = LL.get_all_params(cla)
                values_net = LL.get_all_param_values(cla, trainable=False)
                keys_updates = [k for k in updates.keys()]
                values_updates = [k.get_value() for k in updates.keys()]
                tmp = [paths, param_arch, param_cost, param_updates, param_train,
                       cost_train_avg, acc_train_avg, cost_test_avg, acc_test_avg,
                       keys_net, values_net, keys_updates, values_updates]
                with open(name_dump, 'wb') as f:
                    cPickle.dump(tmp, f)
            
        if param_train['flag_save_preds']:
            if ((i + 1) % param_train['freq_save_preds']) == 0:
                for j, k in enumerate(ds.test_fwd()):
                    path_dump = os.path.join(paths['preds'], "epoch=%04d" % (i + 1))
                    if not os.path.isdir(path_dump):
                        os.makedirs(path_dump)
                    name_dump = os.path.join(path_dump, ds.names_test[j])
                    tmp = funcs['pred'](*k)
                    scipy.io.savemat(name_dump, {'pred': tmp[0]})
    
    return cost_train_avg, acc_train_avg, cost_test_avg, acc_test_avg
