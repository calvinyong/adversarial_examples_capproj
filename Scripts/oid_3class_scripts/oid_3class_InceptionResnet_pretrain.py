import numpy as np

from keras.applications import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Activation, Dense
from keras.utils import to_categorical

epochs = 2

#############
# Functions #
#############

def load_oid3class_data(normalize=True, onehot=True):
    """
    Bird: 0
    Cat: 1
    Fish: 2
    """
    x_test = np.load('../Data/oid_3class_test.npy')
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

train_dir = '../Data/oid_3class/train/'

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

base = InceptionResNetV2(include_top=False, pooling='avg')
x = Dense(3)(base.output)
x = Activation('softmax', name='Prediction')(x)
model = Model(inputs=base.input, outputs=x)

for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True

print(model.summary())
print([layer.trainable for layer in model.layers])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

x_test, y_test = load_oid3class_data()

history = model.fit_generator(train_gen, validation_data=(x_test, y_test),
                              epochs=epochs, steps_per_epoch=60, workers=8)
model.save('oid_3class_InceptionResnet_pretrain.h5')
