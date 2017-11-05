# system modules
import os
import time
import h5py

# numerical modules
import numpy as np
import scipy.sparse as sp


# imports matlab files in python
def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containing the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out


# imports labels from matlab to python and converts them to the python indexing
# (labels start with 0 instead of 1)
def load_labels(fname):
    tmp = load_matlab_file(fname, 'labels')
    tmp -= 1.0
    assert(np.min(tmp) == 0.0)
    tmp = tmp.astype(np.int32)
    return tmp.flatten()


# classification dataset
class dataset_class(object):
    def __init__(self, files_train, files_test, path_descs, path_patch_ops, path_labels, epoch_size=100):
        
        # train / test instances
        self.files_train = files_train
        self.files_test = files_test
        # path to (pre-computed) descriptors
        self.path_descs = path_descs
        # path to (pre-computed) patch operators
        self.path_patch_ops = path_patch_ops
        # path to labels
        self.path_labels = path_labels
        # epoch size
        self.epoch_size = epoch_size
        
        # loading train / test names
        with open(self.files_train, 'r') as f:
            self.names_train = [line.rstrip() for line in f]
        with open(self.files_test, 'r') as f:
            self.names_test = [line.rstrip() for line in f]
        
        # loading the descriptors
        self.descs_train = []
        print("[i] loading train descs... "),
        tic = time.time()
        for name in self.names_train:
            tmp = load_matlab_file(os.path.join(self.path_descs, name), 'desc')
            self.descs_train.append(tmp.astype(np.float32))
        print("%02.2fs" % (time.time() - tic))
        self.descs_test = []
        print("[i] loading test  descs... "),
        tic = time.time()
        for name in self.names_test:
            tmp = load_matlab_file(os.path.join(self.path_descs, name), 'desc')
            self.descs_test.append(tmp.astype(np.float32))
        print("%02.2fs" % (time.time() - tic))
        
        # loading the patch operators
        self.patch_ops_train = []
        print("[i] loading train patch operators... "),
        tic = time.time()
        for name in self.names_train:
            M = load_matlab_file(os.path.join(self.path_patch_ops, name), 'M')
            self.patch_ops_train.append(M.astype(np.float32))
        print("%02.2fs" % (time.time() - tic))
        self.patch_ops_test = []
        print("[i] loading test  patch operators... "),
        tic = time.time()
        for name in self.names_test:
            M = load_matlab_file(os.path.join(self.path_patch_ops, name), 'M')
            self.patch_ops_test.append(M.astype(np.float32))
        print("%02.2fs" % (time.time() - tic))
        
        # loading the labels
        self.labels_train = []
        print("[i] loading train labels... "),
        tic = time.time()
        for name in self.names_train:
            self.labels_train.append(load_labels(os.path.join(self.path_labels, name)))
        print("%02.2fs" % (time.time() - tic))
        self.labels_test = []
        print("[i] loading test  labels... "),
        tic = time.time()
        for name in self.names_test:
            self.labels_test.append(load_labels(os.path.join(self.path_labels, name)))
        print("%02.2fs" % (time.time() - tic))

    def train_iter(self):
        idxs = np.random.permutation(len(self.names_train))
        for i in xrange(self.epoch_size):
            idx = idxs[np.mod(i, len(self.names_train))]
            yield (self.descs_train[idx], self.patch_ops_train[idx], self.labels_train[idx])
    
    # previous version:
    # def train_iter(self):
    #     for i in xrange(self.epoch_size):
    #         idx = np.random.permutation(len(self.names_train))[0]
    #         yield (self.descs_train[idx], self.patch_ops_train[idx], self.labels_train[idx])
    
    def train_fwd(self):
        for idx in xrange(len(self.names_train)):
            yield (self.descs_train[idx], self.patch_ops_train[idx], self.labels_train[idx])
            
    def test_fwd(self):
        for idx in xrange(len(self.names_test)):
            yield (self.descs_test[idx], self.patch_ops_test[idx], self.labels_test[idx])
