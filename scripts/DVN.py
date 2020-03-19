from __future__ import print_function
import numpy as np
import SegmentationNetworkBasis.NetworkBasis.loss  as ls

class FCN(object):
    """ Implementation of DeepVesselNet-FCN with capability of using 3D/2D kernels with or with cross-hair filters"""
    def __init__(self, nchannels=1, nlabels=2, cross_hair=False, dim=3, activation='tanh', levels=None, **kwargs):
        """ Builds the network structure  based on the parameters provided
        parameters:
            nchannels (int) : the number of input channels to the network.
            nlabels (int)   : the number of labels to be predicted.
            cross_hair (bool): whether to use cross hair filters or classical full convolutions (Read Tetteh et al.)
            dim (int)       : the dimension of the network whether its 2D or 3D options are [2, 3] for 2D and 3D versions respectively
            activation (keras supported activation)      : the activation function used in the layers of the network
        """
        if 'loading' in kwargs:
            super(FCN, self).__init__(**kwargs)
            return

        inputs = {'main_input': {'shape': (nchannels,) + (None,)*dim, 'dtype': 'float32'}}
        layers = []
        if levels is None:
            levels = [
                {'filters': 5, 'kernel': 3},
                {'filters': 10, 'kernel': 5},
                {'filters': 20, 'kernel': 5},
                {'filters': 50, 'kernel': 3},
            ]

        conv = 'Conv3DCH' if cross_hair else 'Conv3D'
        if dim==2:
            conv = 'Conv2DCH' if cross_hair else 'Conv2D'

        curinputs = 'main_input'
        cnt = 0

        for level in levels:
            layers.append({
                'layer': conv,
                'inputs': curinputs,
                'sort': -cnt,
                'params': {
                    'name': 'level_'+str(cnt),
                    'filters': level['filters'],
                    'kernel_size': (level['kernel'],)*dim,
                    'strides': (1,)*dim,
                    'padding': 'same',
                    'activation': activation,
                    'data_format': 'channels_first'
                }
            })
            curinputs = 'level_'+str(cnt)
            cnt += 1

        layers.append({
            'layer': 'Conv3D' if dim == 3 else 'Conv2D',
            'inputs': curinputs,
            'sort': -(cnt+1),
            'params': {
                'name': 'presoftmax',
                'filters': nlabels,
                'kernel_size': (1,)*dim,
                'strides': (1,)*dim,
                'padding': 'same',
                'activation': 'linear',
                'data_format': 'channels_first'
                }
        })
        layers.append({
            'layer': 'Softmax',
            'inputs': 'presoftmax',
            'sort': -(cnt + 2),
            'params': {
                'name': 'output',
                'axis': 1
            }
        })
        layers = sorted(layers, key=lambda i: i['sort'], reverse=True)
        models = {'default': {'inputs': 'main_input', 'outputs': 'output'}}
        kwargs['layers'] = layers
        kwargs['input_shapes'] = inputs
        kwargs['models'] = models

    def compile(self, loss=ls.categorical_cross_entropy_loss, optimizer='sgd', metrics=['acc'], **kwargs):
        super(FCN, self).compile(models={'default': {'loss': loss, 'optimizer': optimizer, 'metrics': metrics}})

    def fit(self, **kwargs):
        return super(FCN, self).fit(model='default', **kwargs)

    def fit_generator(self, **kwargs):
        return super(FCN, self).fit_generator(model='default', **kwargs)

    def predict_generator(self, **kwargs):
        return super(FCN, self).predict_generator(model='default', **kwargs)

    def predict(self, **kwargs):
        return super(FCN, self).predict(model='default', **kwargs)

    def evaluate(self, **kwargs):
        return super(FCN, self).evaluate(model='default', **kwargs)


if __name__ == '__main__':
    dim = 2
    net = FCN(cross_hair=True, dim=dim)
    net.compile(loss=ls.weighted_categorical_crossentropy_with_fpr_loss())
    N = (10, 1,) +(64,)*dim
    X = np.random.random(N)
    Y = np.random.randint(2, size=N)
    Y = np.squeeze(Y)