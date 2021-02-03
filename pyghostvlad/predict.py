from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np

#sys.path.append('../tool')
from pyghostvlad.tool import toolkits
import pyghostvlad.utils as ut
import pyghostvlad.model as model
import librosa
from scipy.spatial.distance import pdist, squareform



def configure_model(model_gvlad):

    netparams={
        'gpu': '0',
        'batch_size': 16,
        'net': 'resnet34s',
        'ghost_cluster': 2,
        'vlad_cluster': 8,
        'bottleneck_dim':512,
        'aggregation_mode':'gvlad',
    # set up learning rate, training loss and optimizer.
        'loss':'softmax',
        'test_type':'normal'
    }

    # gpu configuration
    toolkits.initialize_GPU(netparams)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }


    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=netparams)

    # ==> load pre-trained model ???
    if model_gvlad:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(model_gvlad):
            network_eval.load_weights(model_gvlad, by_name=True)
            print('==> successfully loading model {}.'.format(model_gvlad))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(model_gvlad))
    else:
        raise IOError('==> please type in the model to load')

    return network_eval, params, netparams


def predict_features(wav, fs, network_eval, params):

    print('==> start extraction.')

    if int(fs) != 16000:
        wav = librosa.resample(wav, fs, 16000)

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.

    specs = ut.extract_spec(wav, win_length=params['win_length'], sr=params['sampling_rate'],
                         hop_length=params['hop_length'], n_fft=params['nfft'],
                         spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)

    v = network_eval.predict(specs)

    return v

def compute_distance(v1, v2):
    '''
        compute cosine distance
        v1: ndarrau (1,n)
        v2: ndarray (1,n)
    '''
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return (1-(v1 @ v2.T)/(norm_v1*norm_v2)).squeeze()


def compute_distmatrix(X):
    '''
        Parameters
                X: ndarray
                An m by n array of m original observations in an n-dimensional space.
    '''
    D = squareform(pdist(X, 'cosine'))
    return D

if __name__ == "__main__":

    filename = '/mnt/HD2T/BaseDados/VoxCeleb/voxceleb1/vox1_test_wav/wav/id10270/5r0dWxy17C8/00001.wav'
    y,fs = librosa.load(filename,sr=16000, mono=True)
    network_eval, params, netparams = configure_model()
    v1 = predict_features(y, fs, network_eval, params)

    filename2 = '/mnt/HD2T/BaseDados/VoxCeleb/voxceleb1/vox1_dev_wav/wav/id10009/7hpSiT9_gCE/00001.wav'
    y,fs = librosa.load(filename2,sr=16000, mono=True)
    network_eval, params, netparams = configure_model()
    v2 = predict_features(y, fs, network_eval, params)

    print(v1.shape, v2.shape)
    X = np.concatenate((v1,v2),axis=0)

    print(compute_distance(v1, v2))
    print(compute_distmatrix(X))