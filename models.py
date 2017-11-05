# system modules
import os
import time
import logging

# Theano related modules
import theano
import theano.tensor as T
import theano.sparse as Tsp

# Lasagne related modules
import lasagne as L
import lasagne.init as LI
import lasagne.layers as LL
import lasagne.updates as LU
import lasagne.objectives as LO
import lasagne.regularization as LR
import lasagne.nonlinearities as LN

# 
import custom_layers as CL


def categorical_crossentropy_stable(predictions, targets):
    epsilon = 1e-08
    predictions = T.clip(predictions, epsilon, 1.0 - epsilon)
    return T.nnet.categorical_crossentropy(predictions, targets)


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets):
    # categorical_crossentropy_logdomain(log_predictions, targets, nclasses):
    # return -T.sum(theano.tensor.extra_ops.to_one_hot(targets, nclasses) * log_predictions, axis=1)
    # http://deeplearning.net/tutorial/logreg.html#logreg
    return -log_predictions[T.arange(targets.shape[0]), targets]


def arch_class_00(dim_desc, dim_labels, param_arch, logger):
    logger.info('Architecture:')
    # input layers
    desc = LL.InputLayer(shape=(None, dim_desc))
    patch_op = LL.InputLayer(input_var=Tsp.csc_fmatrix('patch_op'), shape=(None, None))
    logger.info('   input  : dim = %d' % dim_desc)
    # layer 1: dimensionality reduction to 16
    n_dim = 16
    net = LL.DenseLayer(desc, n_dim)
    logger.info('   layer 1: FC%d' % n_dim)
    # layer 2: anisotropic convolution layer with 16 filters
    n_filters = 16
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 2: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net) 
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 3: anisotropic convolution layer with 32 filters
    n_filters = 32
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 3: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 4: anisotropic convolution layer with 64 filters
    n_filters = 64
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 4: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)     
    # layer 5: fully connected layer with 256 filters
    n_dim = 256
    net = LL.DenseLayer(net, n_dim)    
    string = '   layer 5: FC%d' % n_dim
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)    
    # layer 6: softmax layer producing a probability on the labels 
    if param_arch['flag_nonlinearity'] == 'softmax':
        cla = LL.DenseLayer(net, dim_labels, nonlinearity=LN.softmax)
        string = '   layer 6: softmax'
    elif param_arch['flag_nonlinearity'] == 'log_softmax':
        cla = LL.DenseLayer(net, dim_labels, nonlinearity=log_softmax)
        string = '   layer 6: log-softmax'
    else:
        raise Exception('[e] the chosen non-linearity is not supported!')
    logger.info(string)
    # outputs
    return desc, patch_op, cla, net, logger


def arch_class_01(dim_desc, dim_labels, param_arch, logger):
    logger.info('Architecture:')
    # input layers
    desc = LL.InputLayer(shape=(None, dim_desc))
    patch_op = LL.InputLayer(input_var=Tsp.csc_fmatrix('patch_op'), shape=(None, None))
    logger.info('   input  : dim = %d' % dim_desc)
    # layer 1: dimensionality reduction to 64
    n_dim = 64
    net = LL.DenseLayer(desc, n_dim)
    logger.info('   layer 1: FC%d' % n_dim)
    # layer 2: anisotropic convolution layer with 64 filters
    n_filters = 64
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 2: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 3: anisotropic convolution layer with 128 filters
    n_filters = 128
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 3: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)      
    # layer 4: anisotropic convolution layer with 256 filters
    n_filters = 256
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 4: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 5: fully connected (linear) layer with 1024 filters and without bias
    n_dim = 1024
    net = LL.DenseLayer(net, n_dim, b=None, nonlinearity=LN.identity)
    string = '   layer 5: linear FC%d' % n_dim
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 6: fully connected layer with 512 filters
    n_dim = 512
    net = LL.DenseLayer(net, n_dim)    
    string = '   layer 6: FC%d' % n_dim
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 7: softmax layer producing a probability on the labels
    if param_arch['non_linearity'] == 'softmax':
        cla = LL.DenseLayer(net, dim_labels, nonlinearity=LN.softmax)
        string = '   layer 7: softmax'
    elif param_arch['non_linearity'] == 'log_softmax':
        cla = LL.DenseLayer(net, dim_labels, nonlinearity=log_softmax)
        string = '   layer 7: log-softmax'
    else:
        raise Exception('[e] the chosen non-linearity is not supported!')
    logger.info(string)
    # outputs
    return desc, patch_op, cla, net, logger


def arch_class_02(dim_desc, dim_labels, param_arch, logger):
    logger.info('Architecture:')
    # input layers
    desc = LL.InputLayer(shape=(None, dim_desc))
    patch_op = LL.InputLayer(input_var=Tsp.csc_fmatrix('patch_op'), shape=(None, None))
    logger.info('   input  : dim = %d' % dim_desc)
    # layer 1: dimensionality reduction to 16
    n_dim = 16
    net = LL.DenseLayer(desc, n_dim)
    logger.info('   layer 1: FC%d' % n_dim)
    # layer 2: anisotropic convolution layer with 16 filters
    n_filters = 16
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 2: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 3: anisotropic convolution layer with 32 filters
    n_filters = 32
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 3: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)      
    # layer 4: anisotropic convolution layer with 64 filters
    n_filters = 64
    net = CL.GCNNLayer([net, patch_op], n_filters, nrings=5, nrays=16)
    string = '   layer 4: IC%d' % n_filters
    if param_arch['flag_batchnorm'] is True:
        net = LL.batch_norm(net)
        string = string + ' + batch normalization'
    logger.info(string)
    # layer 5: softmax layer producing a probability on the labels 
    if param_arch['non_linearity'] == 'softmax':
        cla = LL.DenseLayer(net, dim_labels, nonlinearity=LN.softmax)
        string = '   layer 5: softmax'
    elif param_arch['non_linearity'] == 'log_softmax':
        cla = LL.DenseLayer(net, dim_labels, nonlinearity=log_softmax)
        string = '   layer 5: log-softmax'
    else:
        raise Exception('[e] the chosen non-linearity is not supported!')
    logger.info(string)
    # outputs
    return desc, patch_op, cla, net, logger


def model_class(ds, paths, param_arch, param_cost, param_updates, param_train):

    # create a log file containing the architecture configuration
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger('log_config')
    if 'start_from_epoch' in param_train:
        name_tmp = 'config_from_epoch=%04d.log' % (param_train['start_from_epoch'])
    else:
        name_tmp = 'config.log'
    path_tmp = os.path.join(paths['exp'], name_tmp)
    if not os.path.isfile(path_tmp):
        handler = logging.FileHandler(path_tmp, mode='w')  # to append at the end of the file use: mode='a'
    else:
        raise Exception('[e] the log file ', name_tmp, ' already exists!')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # input dimensions
    dim_desc = ds.descs_train[0].shape[1]
    dim_labels = ds.labels_train[0].shape[0]
    print(dim_labels)

    # architecture definition:
    print("[i] architecture definition... "),
    tic = time.time()
    if param_arch['type'] == 0:
        desc, patch_op, cla, net, logger = arch_class_00(dim_desc, dim_labels, param_arch, logger)
    elif param_arch['type'] == 1:
        desc, patch_op, cla, net, logger = arch_class_01(dim_desc, dim_labels, param_arch, logger)
    elif param_arch['type'] == 2:
        desc, patch_op, cla, net, logger = arch_class_02(dim_desc, dim_labels, param_arch, logger)
    else:
        raise Exception('[e] architecture not supported!')
    print("%02.2fs" % (time.time() - tic))

    # cost function definition:
    print("[i] cost function definition... "),
    tic = time.time()
    pred = LL.get_output(cla, deterministic=True)  # in case we use dropout
    feat = LL.get_output(net)
    target = T.ivector('target')
    # data term
    if param_cost['cost_func'] == 'cross_entropy':
        if param_arch['non_linearity'] == 'softmax':
            cost_dataterm = T.mean(LO.categorical_crossentropy(pred, target))  # in the original code we were using *.mean() instead of T.mean(*)
        elif param_arch['non_linearity'] == 'log_softmax':
            cost_dataterm = T.mean(categorical_crossentropy_logdomain(pred, target))
    elif param_cost['cost_func'] == 'cross_entropy_stable':
        if param_arch['non_linearity'] == 'softmax':
            cost_dataterm = T.mean(categorical_crossentropy_stable(pred, target))
        else:
            raise Exception('[e] the chosen cost function is not implemented for the chosen non-linearity!')
    else:
        raise Exception('[e] the chosen cost function is not supported!')
    # classification accuracy
    acc = LO.categorical_accuracy(pred, target).mean()
    # regularization
    cost_reg = param_cost['mu'] * LR.regularize_network_params(cla, LR.l2)
    # cost function
    cost = cost_dataterm + cost_reg
    # get params
    params = LL.get_all_params(cla)
    # gradient definition
    grad = T.grad(cost, params)
    grad_norm = T.nlinalg.norm(T.concatenate([g.flatten() for g in grad]), 2)
    print("%02.2fs" % (time.time() - tic))

    # updates definition:
    print("[i] gradient updates definition... "),
    tic = time.time()
    if param_updates['method'] == 'momentum':
        if param_updates.get('learning_rate') is not None:
            learning_rate = param_updates['learning_rate']  # default: 1.0
        else:
            raise Exception('[e] missing learning_rate parameter!')
        if param_updates.get('momentum') is not None:
            momentum = param_updates['momentum']  # default: 0.9
        else:
            raise Exception('[e] missing learning_rate parameter!')
        updates = LU.momentum(grad, params, learning_rate, momentum)
    elif param_updates['method'] == 'adagrad':
        if param_updates.get('learning_rate') is not None:
            learning_rate = param_updates['learning_rate']  # default: 1.0
        else:
            raise Exception('[e] missing learning_rate parameter!')
        updates = LU.adagrad(grad, params, learning_rate)
    elif param_updates['method'] == 'adadelta':
        if param_updates.get('learning_rate') is not None:
            learning_rate = param_updates['learning_rate']  # default: 1.0
        else:
            raise Exception('[e] missing learning_rate parameter!')
        updates = LU.adadelta(grad, params, learning_rate)
    elif param_updates['method'] == 'adam':
        if param_updates.get('learning_rate') is not None:
            learning_rate = param_updates['learning_rate']  # default: 1e-03
        else:
            raise Exception('[e] missing learning_rate parameter!')
        if param_updates.get('beta1') is not None:
            beta1 = param_updates['beta1']  # default: 0.9
        else:
            raise Exception('[e] missing beta1 parameter!')
        if param_updates.get('beta2') is not None:
            beta2 = param_updates['beta2']  # default: 0.999
        else:
            raise Exception('[e] missing beta2 parameter!')
        if param_updates.get('epsilon') is not None:
            epsilon = param_updates['epsilon']  # default: 1e-08
        else:
            raise Exception('[e] missing epsilon parameter!')
        updates = LU.adam(grad, params, learning_rate, beta1, beta2, epsilon)
    else:
        raise Exception('[e] updates method not supported!')
    print("%02.2fs" % (time.time() - tic))

    # train / test functions:
    funcs = dict()
    print("[i] compiling function 'train'... "),
    tic = time.time()
    funcs['train'] = theano.function([desc.input_var, patch_op.input_var, target],
                                     [cost, cost_dataterm, cost_reg, grad_norm, acc],
                                     updates=updates,
                                     allow_input_downcast=True,
                                     on_unused_input='warn')
    print("%02.2fs" % (time.time() - tic))
    print("[i] compiling function 'fwd'... "),
    tic = time.time()
    funcs['fwd'] = theano.function([desc.input_var, patch_op.input_var, target], 
                                   [cost, grad_norm, acc],
                                   allow_input_downcast=True,
                                   on_unused_input='ignore')
    print("%02.2fs" % (time.time() - tic))
    print("[i] compiling function 'pred'... "),
    tic = time.time()
    funcs['pred'] = theano.function([desc.input_var, patch_op.input_var, target], 
                                    [pred],
                                    allow_input_downcast=True,
                                    on_unused_input='ignore')
    print("%02.2fs" % (time.time() - tic))
    print("[i] compiling function 'feat'... "),
    tic = time.time()
    funcs['feat'] = theano.function([desc.input_var, patch_op.input_var, target], 
                                    [feat],
                                    allow_input_downcast=True,
                                    on_unused_input='ignore')
    print("%02.2fs" % (time.time() - tic))
    
    # save cost function parameters to a config file
    logger.info('\nCost function parameters:')
    logger.info('   cost function = %s' % param_cost['cost_func'])
    logger.info('   mu            = %e' % param_cost['mu'])

    # save updates parameters to a config file
    logger.info('\nUpdates parameters:')
    logger.info('   method        = %s' % param_updates['method'])
    logger.info('   learning rate = %e' % param_updates['learning_rate'])
    if param_updates['method'] == 'momentum':
        logger.info('   momentum      = %e' % param_updates['momentum'])
    if param_updates['method'] == 'adam':
        logger.info('   beta1         = %e' % param_updates['beta1'])
        logger.info('   beta2         = %e' % param_updates['beta2'])
        logger.info('   epsilon       = %e' % param_updates['epsilon'])

    # save training parameters to a config file
    logger.info('\nTraining parameters:')
    logger.info('   epoch size = %d' % ds.epoch_size)
    
    return funcs, cla, updates
