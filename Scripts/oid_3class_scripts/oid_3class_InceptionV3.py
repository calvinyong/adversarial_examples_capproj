import numpy as np

from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import to_categorical
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

epochs = 200

#############
# Functions #
#############

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def load_oid3class_data(normalize=True, onehot=True):
    """
    Bird: 0
    Cat: 1
    Fish: 2
    """
    x_test = np.load('oid_3class_test.npy')
    if normalize:
        x_test = x_test / 255.0
    x_test = x_test.astype('float32')
    
    y_test = np.zeros(750)
    y_test[250:500] = 1
    y_test[500:] = 2
    if onehot:
        y_test = to_categorical(y_test)
        
    return x_test, y_test

########
# Main #
########

train_dir = '../Dataset/train/'

train_idg = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)

train_gen = train_idg.flow_from_directory(train_dir,
                                          batch_size=25,
                                          target_size=(299, 299))

model = InceptionV3(classes=3, pooling='avg', weights=None)
print(model.summary())

model_type = 'inceptionV3'
save_dir = os.path.join(os.getcwd(), 'inceptionV3_saved_models')
model_name = 'oid_3class_%s_{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
csvlog = CSVLogger('oid_3class_InceptionV3_log.csv')

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [checkpoint, csvlog, lr_reducer, lr_scheduler]

model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy',
                             metrics=['acc'])

x_test, y_test = load_oid3class_data()

history = model.fit_generator(train_gen, validation_data=(x_test, y_test), epochs=epochs,
                              steps_per_epoch=60, workers=8, callbacks=callbacks)
