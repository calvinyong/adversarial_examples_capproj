import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

bird_imgs_train = glob('../Data/bird_or_bicycle/0.0.3/train/bird/*')
bicycle_imgs_train = glob('../Data/bird_or_bicycle/0.0.3/train/bicycle/*')

bird_imgs_test = glob('../Data/bird_or_bicycle/0.0.3/test/bird/*')
bicycle_imgs_test = glob('../Data/bird_or_bicycle/0.0.3/test/bicycle/*')


array = np.empty(shape=(0, 299, 299, 3), dtype=int)

for f in tqdm(bird_imgs_train, desc='bird train'):
    img = cv2.imread(f)
    img = np.expand_dims(img, axis=0)
    array = np.append(array, img, axis=0)

for f in tqdm(bicycle_imgs_train, desc='bicycle train'):
    img = cv2.imread(f)
    img = np.expand_dims(img, axis=0)
    array = np.append(array, img, axis=0)

array = array[:, :, :, ::-1].copy()
np.save('bird_bicycle_train.npy', array)

print("Train saved ndarray of shape:", array.shape)


array = np.empty(shape=(0, 299, 299, 3), dtype=int)

for f in tqdm(bird_imgs_test, desc='bird test'):
    img = cv2.imread(f)
    img = np.expand_dims(img, axis=0)
    array = np.append(array, img, axis=0)

for f in tqdm(bicycle_imgs_test, desc='bicycle test'):
    img = cv2.imread(f)
    img = np.expand_dims(img, axis=0)
    array = np.append(array, img, axis=0)

array = array[:, :, :, ::-1].copy()
np.save('bird_bicycle_test.npy', array)

print("Test saved ndarray of shape:", array.shape)
