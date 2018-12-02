import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from cleverhans.attacks import FastGradientMethod, DeepFool, MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K

import os
import sys
sys.path.append('../Capstone_project/Adversarial_examples_capproj/Final_notebook')
import proj_utils as utils

os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.set_learning_phase(0)

def batches(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Data
x_test, y_test = utils.load_oid3class_data('oid_3class_test.npy')

# Models
xception = load_model('Saved_models/good_models/oid_3class/oid_3class_xception_083.h5')
inceptionV3 = load_model('Saved_models/good_models/oid_3class/oid_3class_inceptionV3_098.h5')
inceptionResnet = load_model('Saved_models/pretrained_models/oid_3class_InceptionResnet_pretrain.h5')
xception.name = 'Xception'
inceptionV3.name = 'InceptionV3'
inceptionResnet.name = 'InceptionResnetV2'

row_names = ["Xception", "InceptionV3", "InceptionResnetV2"]

models = [xception, inceptionV3, inceptionResnet]
eps_list = [1.0/255, 3.0/255, 8.0/255, 13.0/255]
for eps in eps_list:
    fgsm_df = pd.DataFrame()
    L1_avgs = []
    L2_avgs = []
    Linf_avgs = []
    for model in tqdm(models, desc='Model'):
        ch_model = KerasModelWrapper(model)
        fgsm = FastGradientMethod(ch_model, sess=K.get_session())
        fgsm_params = {'eps': eps,
                       'clip_min': 0.0,
                       'clip_max': 1.0}
        
        # Batch to avoid ResourceExhaustedError
        adv = []
        for batch in tqdm(batches(x_test, 10), desc='Make adv'):
            adv.append(fgsm.generate_np(batch, **fgsm_params))
        adv = np.concatenate(adv)

        # Get loss/acc for adversarial examples
        test_loss_acc_adv = [xception.evaluate(adv, y_test),
                            inceptionV3.evaluate(adv, y_test),
                            inceptionResnet.evaluate(adv, y_test)]
        
        # Debug
        print(model.name)
        print(test_loss_acc_adv)
        
        test_loss_adv = [x[0] for x in test_loss_acc_adv]
        test_acc_adv = [x[1] for x in test_loss_acc_adv]
        
        L1 = []
        L2 = []
        Linf = []
        for i in range(x_test.shape[0]):
            vector = (adv[i] - x_test[i]).copy()
            vector = vector.flatten()
            L1.append(np.linalg.norm(vector, ord=1))
            L2.append(np.linalg.norm(vector))
            Linf.append(np.linalg.norm(vector, ord=np.inf))
        L1 = np.array(L1)
        L2 = np.array(L2)
        Linf = np.array(Linf)

        L1_avgs.append(L1.mean())
        L2_avgs.append(L2.mean())
        Linf_avgs.append(Linf.mean())

        # Make df
        # use list instead of dict to preserve order
        tmp = pd.DataFrame([test_loss_adv, test_acc_adv], columns=row_names, index=[(model.name, 'Loss'), (model.name, 'Acc')]).T
        #tmp.columns = pd.MultiIndex.from_tuples(tmp.columns)
        fgsm_df = pd.concat([fgsm_df, tmp], axis=1)
    L_df = pd.DataFrame([L1_avgs, L2_avgs, Linf_avgs], columns=row_names, index=['L1', 'L2', 'L_inf']).T
    fgsm_df = pd.concat([L_df, fgsm_df], axis=1)
    fgsm_df.to_csv('csvs/oid3class_fgsm_' + str(eps) + ".csv")




deepfool_df = pd.DataFrame()
L1_avgs = []
L2_avgs = []
Linf_avgs = []
for model in tqdm(models, desc='Model'):
    ch_model = KerasModelWrapper(model)
    deepfool = DeepFool(ch_model, sess=K.get_session())
    deepfool_params = {'nb_candidate': 3}
    
    # Batch to avoid ResourceExhaustedError
    adv = []
    for batch in tqdm(batches(x_test, 10), desc='Make adv'):
        adv.append(deepfool.generate_np(batch, **deepfool_params))
    adv = np.concatenate(adv)

    # Get loss/acc for adversarial examples
    test_loss_acc_adv = [xception.evaluate(adv, y_test),
                        inceptionV3.evaluate(adv, y_test),
                        inceptionResnet.evaluate(adv, y_test)]
    
    # Debug
    print(model.name)
    print(test_loss_acc_adv)
    
    test_loss_adv = [x[0] for x in test_loss_acc_adv]
    test_acc_adv = [x[1] for x in test_loss_acc_adv]
    
    L1 = []
    L2 = []
    Linf = []
    for i in range(x_test.shape[0]):
        vector = (adv[i] - x_test[i]).copy()
        vector = vector.flatten()
        L1.append(np.linalg.norm(vector, ord=1))
        L2.append(np.linalg.norm(vector))
        Linf.append(np.linalg.norm(vector, ord=np.inf))
    L1 = np.array(L1)
    L2 = np.array(L2)
    Linf = np.array(Linf)

    L1_avgs.append(L1.mean())
    L2_avgs.append(L2.mean())
    Linf_avgs.append(Linf.mean())

    # Make df
    # use list instead of dict to preserve order
    tmp = pd.DataFrame([test_loss_adv, test_acc_adv], columns=row_names, index=[(model.name, 'Loss'), (model.name, 'Acc')]).T
    #tmp.columns = pd.MultiIndex.from_tuples(tmp.columns)
    deepfool_df = pd.concat([deepfool_df, tmp], axis=1)
L_df = pd.DataFrame([L1_avgs, L2_avgs, Linf_avgs], columns=row_names, index=['L1', 'L2', 'L_inf']).T
deepfool_df = pd.concat([L_df, deepfool_df], axis=1)
deepfool_df.to_csv('csvs/oid3class_deepfool.csv')


eps_list = [1.0/255, 3.0/255, 8.0/255, 13.0/255]
for eps in eps_list:
    madry_df = pd.DataFrame()
    L1_avgs = []
    L2_avgs = []
    Linf_avgs = []
    for model in tqdm(models, desc='Model'):
        ch_model = KerasModelWrapper(model)
        madry = MadryEtAl(ch_model, sess=K.get_session())
        madry_params = {'eps': eps,
                       'clip_min': 0.0,
                       'clip_max': 1.0}
        
        # Batch to avoid ResourceExhaustedError
        adv = []
        for batch in tqdm(batches(x_test, 10), desc='Make adv'):
            adv.append(madry.generate_np(batch, **fgsm_params))
        adv = np.concatenate(adv)

        # Get loss/acc for adversarial examples
        test_loss_acc_adv = [xception.evaluate(adv, y_test),
                            inceptionV3.evaluate(adv, y_test),
                            inceptionResnet.evaluate(adv, y_test)]
        
        # Debug
        print(model.name)
        print(test_loss_acc_adv)
        
        test_loss_adv = [x[0] for x in test_loss_acc_adv]
        test_acc_adv = [x[1] for x in test_loss_acc_adv]
        
        L1 = []
        L2 = []
        Linf = []
        for i in range(x_test.shape[0]):
            vector = (adv[i] - x_test[i]).copy()
            vector = vector.flatten()
            L1.append(np.linalg.norm(vector, ord=1))
            L2.append(np.linalg.norm(vector))
            Linf.append(np.linalg.norm(vector, ord=np.inf))
        L1 = np.array(L1)
        L2 = np.array(L2)
        Linf = np.array(Linf)

        L1_avgs.append(L1.mean())
        L2_avgs.append(L2.mean())
        Linf_avgs.append(Linf.mean())

        # Make df
        # use list instead of dict to preserve order
        tmp = pd.DataFrame([test_loss_adv, test_acc_adv], columns=row_names, index=[(model.name, 'Loss'), (model.name, 'Acc')]).T
        #tmp.columns = pd.MultiIndex.from_tuples(tmp.columns)
        madry_df = pd.concat([madry_df, tmp], axis=1)
    L_df = pd.DataFrame([L1_avgs, L2_avgs, Linf_avgs], columns=row_names, index=['L1', 'L2', 'L_inf']).T
    madry_df = pd.concat([L_df, madry_df], axis=1)
    madry_df.to_csv('csvs/oid3class_madry_' + str(eps) + ".csv")

