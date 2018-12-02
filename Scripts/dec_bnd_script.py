from pathlib import Path
import proj_utils as utils
import os

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="1"

K.set_learning_phase(0)
(cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = cifar10.load_data()

# Normalize and one-hot encode
cifar_x_test = cifar_x_test / 255.0
cifar_y_test_tmp = np.squeeze(cifar_y_test)
cifar_y_test = to_categorical(cifar_y_test)

# Models for CIFAR10
deepcnn = load_model('Saved_models/good_models/cifar10/cifar10_vgg_like_175.h5')
resnet20 = load_model('Saved_models/good_models/cifar10/cifar10_ResNet20v2_model.176.h5')
resnet56 = load_model('Saved_models/good_models/cifar10/cifar10_ResNet56v2_model.136.h5')
densenet = load_model('Saved_models/good_models/cifar10/cifar10_densenet121_083.h5')
deepcnn.name = 'Deep_CNN'
resnet20.name = 'ResNet20'
resnet56.name = 'ResNet56'
densenet.name = 'DenseNet121'
cifar10_models = [deepcnn, resnet20, resnet56, densenet]

ind = [7733, 4337, 5404, 5825, 5863, 421, 7629, 6807, 1, 11]
cifar_sample_img = cifar_x_test[ind].copy()
cifar_sample_img_label = cifar_y_test[ind].copy()
del cifar_x_test
del cifar_y_test

seeds = [42, 369, 1818, 2093, 4937, 5285, 7071, 7817, 7967, 8367]

for model in cifar10_models:
    print('---------' + model.name + '---------')
    for i in range(10):
        label = utils.CIFAR10_LABEL_NAMES[i]
        print('---------' + label + '---------')
        dirpath  = 'decision_boundary_pics/' + model.name + '/' + label + '/'
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            dr_params = dict(bounds=(-10, 10), num=1000, seed=seed, cmap='tab10', figsize=(16, 9))
            try:
                fig, ax = utils.decision_regions(cifar_sample_img[i], model,
                                                img_label=cifar_sample_img_label[i], **dr_params)
            except AssertionError:
                continue
            
            ax.set_title("Decision Regions for " + model.name, fontsize=20)
            fig.savefig(dirpath + label + '_' + str(seed) + '.png')
            print(label, "seed", seed, "done")

