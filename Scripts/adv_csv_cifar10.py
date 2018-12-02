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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
K.set_learning_phase(0)

def batches(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_test_tmp = np.squeeze(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Models
vgglike = load_model('Saved_models/good_models/cifar10/cifar10_vgg_like_175.h5')
resnet20 = load_model('Saved_models/good_models/cifar10/cifar10_ResNet20v2_model.176.h5')
resnet56 = load_model('Saved_models/good_models/cifar10/cifar10_ResNet56v2_model.136.h5')
densenet = load_model('Saved_models/good_models/cifar10/cifar10_densenet121_083.h5')
vgglike.name = 'Deep_CNN'
resnet20.name = 'ResNet20'
resnet56.name = 'ResNet56'
densenet.name = 'Densenet121'

train_loss_acc_clean = [vgglike.evaluate(x_train, y_train),
                        resnet20.evaluate(x_train, y_train),
                        resnet56.evaluate(x_train, y_train),
                        densenet.evaluate(x_train, y_train)]
train_loss_clean = [x[0] for x in train_loss_acc_clean]
train_acc_clean = [x[1] for x in train_loss_acc_clean]

test_loss_acc_clean = [vgglike.evaluate(x_test, y_test),
                       resnet20.evaluate(x_test, y_test),
                       resnet56.evaluate(x_test, y_test),
                       densenet.evaluate(x_test, y_test)]
test_loss_clean = [x[0] for x in test_loss_acc_clean]
test_acc_clean = [x[1] for x in test_loss_acc_clean]

row_names = ["Deep_CNN", "ResNet20", "ResNet56", "Densenet121"]
col_names = [('Train', 'Loss'), ('Train', 'Acc'), ('Test', 'Loss'), ('Test', 'Acc')]
df = pd.DataFrame([train_loss_clean, train_acc_clean, test_loss_clean, test_acc_clean], columns=row_names).T
df.to_csv('csvs/cifar_models_loss_acc.csv')


models = [vgglike, resnet20, resnet56, densenet]
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
        for batch in tqdm(batches(x_test, 100), desc='Make adv'):
            adv.append(fgsm.generate_np(batch, **fgsm_params))
        adv = np.concatenate(adv)

        # Get loss/acc for adversarial examples
        test_loss_acc_adv = [vgglike.evaluate(adv, y_test),
                            resnet20.evaluate(adv, y_test),
                            resnet56.evaluate(adv, y_test),
                            densenet.evaluate(adv, y_test)]
        
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
    fgsm_df.to_csv('csvs/cifar10_fgsm_' + str(eps) + ".csv")



deepfool_df = pd.DataFrame()
L1_avgs = []
L2_avgs = []
Linf_avgs = []
for model in tqdm(models, desc='Model'):
    ch_model = KerasModelWrapper(model)
    deepfool = DeepFool(ch_model, sess=K.get_session())
    
    # Batch to avoid ResourceExhaustedError
    adv = []
    for batch in tqdm(batches(x_test, 100), desc='Make adv'):
        adv.append(deepfool.generate_np(batch))
    adv = np.concatenate(adv)

    # Get loss/acc for adversarial examples
    test_loss_acc_adv = [vgglike.evaluate(adv, y_test),
                        resnet20.evaluate(adv, y_test),
                        resnet56.evaluate(adv, y_test),
                        densenet.evaluate(adv, y_test)]
    
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
deepfool_df.to_csv('csvs/cifar10_deepfool.csv')



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
        for batch in tqdm(batches(x_test, 100), desc='Make adv'):
            adv.append(madry.generate_np(batch, **madry_params))
        adv = np.concatenate(adv)

        # Get loss/acc for adversarial examples
        test_loss_acc_adv = [vgglike.evaluate(adv, y_test),
                            resnet20.evaluate(adv, y_test),
                            resnet56.evaluate(adv, y_test),
                            densenet.evaluate(adv, y_test)]
        
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
    madry_df.to_csv('csvs/cifar10_madry_' + str(eps) + ".csv")

