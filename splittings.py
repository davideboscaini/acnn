# system modules
import os


# train / test splitting:
def default_split_FAUST(path_exp):
    #
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)
    # opening writable files
    files_train = open(os.path.join(path_exp, 'files_train.txt'), 'w')
    files_test = open(os.path.join(path_exp, 'files_test.txt'), 'w')
    # first 80 shapes are considered as training set
    for idx in xrange(0, 80):
        files_train.write('tr_reg_%.3d.mat\n' % idx)
    # last 20 shapes are considered as test set
    for idx in xrange(80, 100):
        files_test.write('tr_reg_%.3d.mat\n' % idx)
    # closing files
    files_train.close()
    files_test.close()
