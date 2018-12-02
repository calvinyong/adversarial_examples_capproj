import numpy as np

from keras.preprocessing import image
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.applications import ResNet50, VGG16
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    train_dir = ''
    test_dir =  ''

    base = ResNet50(weights=None, include_top=False, pooling='avg')
    print(base.summary())
    model = Sequential()
    model.add(base)
    model.add(Dense(1))
    model.add(Activation('relu'))
    print(model.summary())