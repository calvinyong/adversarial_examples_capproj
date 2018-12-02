import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

target_shape = (299, 299)

bird_test = glob('../Dataset/test/Bird/*')
cat_test = glob('../Dataset/test/Cat/*')
fish_test = glob('../Dataset/test/Fish/*')

test_list = []

for f in tqdm(bird_test, desc='bird test'):
    img = cv2.imread(f)
    img = cv2.resize(img, target_shape)
    img = np.expand_dims(img, axis=0)
    test_list.append(img)

for f in tqdm(cat_test, desc='cat test'):
    img = cv2.imread(f)
    img = cv2.resize(img, target_shape)
    img = np.expand_dims(img, axis=0)
    test_list.append(img)

for f in tqdm(fish_test, desc='fish test'):
    img = cv2.imread(f)
    img = cv2.resize(img, target_shape)
    img = np.expand_dims(img, axis=0)
    test_list.append(img)

test_array = np.concatenate(test_list)
test_array = test_array[:, :, :, ::-1].copy()
test_array = test_array.astype('uint8').copy()

np.save('oid_3class_test.npy', test_array)
print("Test array saved. Shape is", test_array.shape)
