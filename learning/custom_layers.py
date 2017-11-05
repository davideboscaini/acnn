# Theano related modules
import theano
import theano.tensor as T
import theano.sparse as Tsp

# Lasagne related modules
import lasagne as L
import lasagne.init as LI
import lasagne.layers as LL
import lasagne.objectives as LO
import lasagne.regularization as LR
import lasagne.nonlinearities as LN

# GCNN layer
class GCNNLayer(LL.MergeLayer):
    """
    Geodesic Convolutional Neural Network (GCNN) layer
    """
    def __init__(self, incomings, nfilters, nrings=5, nrays=16,
                 W=LI.GlorotNormal(), b=LI.Constant(0.0),
                 normalize_rings=False, normalize_input=False, take_max=True, 
                 nonlinearity=LN.rectify, **kwargs):
        super(GCNNLayer, self).__init__(incomings, **kwargs)
        
        # patch operator sizes
        self.nfilters = nfilters
        self.nrings = nrings
        self.nrays = nrays
        self.filter_shape = (nfilters, self.input_shapes[0][1], nrings, nrays)
        self.biases_shape = (nfilters, )
        # path operator parameters
        self.normalize_rings = normalize_rings
        self.normalize_input = normalize_input
        self.take_max = take_max
        self.nonlinearity = nonlinearity
        
        # layer parameters:
        # y = Wx + b, where x are the input features and y are the output features
        self.W = self.add_param(W, self.filter_shape, name="W")
        self.b = self.add_param(b, self.biases_shape, name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        shp = input_shapes[0]
        nrays = self.nrays
        if self.take_max:
            nrays = 1
        out_shp = (shp[0], self.nfilters * 1 * nrays)
        return out_shp

    def get_output_for(self, inputs, **kwargs):
        y, M = inputs

        if self.normalize_input:
            y /= T.sqrt(T.sum(T.sqr(y), axis=1) + 1e-5).dimshuffle(0, 'x')

        # theano.dot works both for sparse and dense matrices
        desc_net = theano.dot(M, y)

        desc_net = T.reshape(desc_net, (M.shape[1], self.nrings, self.nrays, y.shape[1]))
        desc_net = desc_net.dimshuffle(0, 3, 1, 2)

        if self.normalize_rings:
            # Unit length per ring
            desc_net /= (1e-5 + T.sqrt(T.sum(T.sqr(desc_net), axis=2) + 1e-5).dimshuffle(0, 1, 'x', 2))

        # pad it along the rays axis so that conv2d produces circular
        # convolution along that dimension
        desc_net = T.concatenate([desc_net, desc_net[:, :, :, :-1]], axis=3)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding
        # OLD VERSION --------------------------------------------------------------------------------------------
        # y = theano.tensor.nnet.conv.conv2d(desc_net, self.W, 
        #     (self.input_shapes[0][0], self.filter_shape[1], self.nrings, self.nrays * 2 - 1), self.filter_shape)
        # --------------------------------------------------------------------------------------------------------
        y = theano.tensor.nnet.conv2d(desc_net, self.W, (self.input_shapes[0][0], self.filter_shape[1], self.nrings, self.nrays * 2 - 1), self.filter_shape)
        
        if self.take_max:
            # take the max activation along all rotations of the disk
            y = T.max(y, axis=3).dimshuffle(0, 1, 2, 'x')
            # y is now shaped as N x outmaps x 1 x 1

        if self.b is not None:
            y += self.b.dimshuffle('x', 0, 'x', 'x')

        y = y.flatten(2)

        return self.nonlinearity(y)


# ACNN layer
class ACNNLayer(GCNNLayer):
    """
    Anisotropic Convolutional Neural Network (ACNN) layer
    """
    def __init__(self, incomings, nfilters, nrings=5, nrays=16,
                 W=LI.GlorotNormal(), b=LI.Constant(0.0),
                 normalize_rings=False, normalize_input=False, take_max=True, 
                 nonlinearity=LN.rectify, **kwargs):
        super(ACNNLayer, self).__init__(incomings, nfilters, nrings, nrays,
                 W, b,
                 normalize_rings, normalize_input, take_max, 
                 nonlinearity, **kwargs)
    # def get_output_shape_for(self, input_shapes):
    #     super(ACNNLayer, self).get_output_shape_for(input_shapes)
    # def get_output_for(self, inputs, **kwargs):
    #     super(ACNNLayer, self).get_output_for(inputs, **kwargs)


# Covariance layer
class COVLayer(LL.Layer):
    def __init__(self, incoming, **kwargs):
        super(COVLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

    def get_output_for(self, input, **kwargs):
        x = input
        x -= x.mean(axis=0)
        x = T.dot(x.T, x) / (self.input_shape[0] - 1)
        x = x.flatten(2)
        return x
