# system modules
import os
#import time
import h5py
import glob

# numerical modules
import numpy as np
import scipy.io

# plotting modules
import seaborn as sns
from seaborn import plt as plt
#%matplotlib inline

#
import datasets


def kim_curve(assignment, gt, D, A, th=0.3):
    """
    assignment: vector of indices. row is index on the source shape
    gt: vector of gt assignment
    D: geodesic distances on the target shape
    """
    err = D[assignment, gt] / np.sqrt(A)
    th = np.linspace(0, th, 100)
    errth = err.reshape(1, -1) <= th.reshape(-1, 1)
    return np.mean(errth, axis=1), th


def compute_area(vertices, faces):
    A = 0.0
    for f in faces:
        A += np.linalg.norm(np.cross(vertices[f[1]]-vertices[f[0]], vertices[f[2]]-vertices[f[0]]))
    return A / 2.0


def compute_kim_curve(path_shapes, path_geods, path_preds, th):
    # computes the average Kim curve for all shapes wrt tr_reg_000:
    # load the target shape tr_reg_000
    shape_target = h5py.File(os.path.join(path_shapes, 'tr_reg_000.mat'), 'r')['shape']
    vertices = np.concatenate([np.asarray(shape_target[i]) for i in ['X', 'Y', 'Z']]).T
    # load its geodesic distances
    D = datasets.load_matlab_file(os.path.join(path_geods, 'tr_reg_000.mat'), 'geods')
    # shapes from FAUST_registrations dataset are aligned, i.e. vertex i on one shape correspond to vertex i on another shape 
    # if another dataset is considered, change the following line
    gt = np.arange(6890)

    tmp = dict()
    tmp['acc'] = []
    tmp['kc'] = []
    
    for i in glob.glob(os.path.join(path_preds, '*.mat')):
        T = scipy.io.loadmat(i)['pred']
        assignment = T.argmax(axis=1)
        A = compute_area(vertices, np.asarray(shape_target['TRIV']).T-1)
        kc = kim_curve(assignment, gt, D, A, th)
        tmp['acc'].append(np.mean(assignment == gt))
        tmp['kc'].append(kc[0])
        #print tmp['acc'][-1]
    
    tmp['kc_avg'] = np.mean([j.reshape(-1, 1) for j in tmp['kc']], axis=0)
    
    return tmp


def plot_kim_curve(tmp):
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})
    sns.set_style("darkgrid")

    plt.figure(figsize=(20, 10))
    plt.hold('on')
    plt.plot(np.linspace(0, 0.3, 100), tmp['kc_avg'])
    plt.ylim([0, 1])

#    plt.figure(figsize=(10,5))
#    plt.hold('on')
#    legend = []
#    for k,v in bench_res.iteritems():
#        plt.plot(np.linspace(0, 0.3, 100), v['kc_avg'])
#        legend.append(k)
#    plt.ylim([0, 1])
#    plt.legend(legend, loc='lower right')
