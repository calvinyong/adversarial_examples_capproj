"""
Functions for the Adversarial Examples notebook
"""

from collections import Counter
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

from keras import backend as K
from keras.utils import to_categorical

#############
# Variables #
#############

CIFAR10_LABEL_NAMES = {0: 'airplane',
                       1: 'automobile',
                       2: 'bird',
                       3: 'cat',
                       4: 'deer',
                       5: 'dog',
                       6: 'frog',
                       7: 'horse',
                       8: 'ship',
                       9: 'truck'}

OID_3CLASS_LABEL_NAMES = {0: 'Bird', 1: 'Cat', 2: 'Fish'}

#############
# Load Data #
#############

def load_image(loc, normalize=True):
    """
    Load an image with cv2, convert to RBG, and 
    normalize
    
    '../Data/bird_or_bicycle/0.0.3/test/bird/3ceee1ba9a1300ef.jpg'
    
    # Arguments
        loc: string
            File location of the image
        normalize: Boolean, default=True
            If True (default), normalize image to [0, 1]
    # Returns
        img: np.array
            The image as a numpy array (channels last)
    """
    img = cv2.imread(loc)[:, :, ::-1]
    if normalize:
        img = img / 255
        img = img.astype('float32')
    return img

def load_oid3class_data(dirloc='../Data/oid_3class_test.npy',
                        normalize=True, onehot=True):
    """
    Load test data for small subset of Open Images Dataset V4.
    The test set was resized to (299, 299) using cv2. Channel
    format is RBG
    
    Bird: 0
    Cat: 1
    Fish: 2
    
    # Arguments
        dirloc: string, default=='../Data/oid_3class_test.npy'
            Location of the OIDv4 3 classes .npy test data
        normalize: Boolean, default=True
            If True (default), normalize image to [0, 1]
        onehot: Boolean, default=True
            If True (default), one-hot encode y_test labels
        
    # Returns
        x_test, y_test: test set
    """
    x_test = np.load(dirloc)
    if normalize:
        x_test = x_test / 255.0
        x_test = x_test.astype('float32')
    
    y_test = np.zeros(750)
    y_test[250:500] = 1
    y_test[500:] = 2
    if onehot:
        y_test = to_categorical(y_test)
        
    return x_test, y_test

#########################
# Toy Example Functions #
#########################

def plot_history(history):
    """
    Plot train/test loss and accuracy curves.
    https://keras.io/visualization/
    
    # Arguments
        history: keras history
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    axs[0].plot(epochs, acc, label='Training acc')
    axs[0].plot(epochs, val_acc, label='Validation acc')
    axs[0].set_title('Training and Validation accuracy')
    axs[0].legend()
    
    axs[1].plot(epochs, loss, label='Training loss')
    axs[1].plot(epochs, val_loss, label='Validation loss')
    axs[1].set_title('Training and Validation loss')
    axs[1].legend()


def plot_decision_boundary(X, y, model, bounds, x_train=None, y_train=None,
                           train_only=False, steps=1000, cmap=None, alpha=0.25,
                           figsize=(8, 6), ax=None):
    """
    Plot the decision regions and boundaries for 2d points
    
    Code based off:
    https://github.com/NSAryan12/nn-from-scratch/blob/master/nn-from-scratch.ipynb
    
    # Arguments
        X: np.array
            Input data
        y: np.array
            One-hot encoded targets
        model: keras model
        bounds: tuple (l, r, d, u)
            x and y axis bounds for the figure
        x_train: np.array, default=None
            Inputs for the training data. Must be supplied if train_only
            is True
        y_train: np.array, default=None
            Targets for training data. Must be supplied if train_only
            is True
        train_only: Boolean, default=False
            If True, plot only the training set. x_train and y_train
            must also be supplied
        steps: int, default=1000
            Number of points between lower bound and upper bound for
            both xaxis and yaxis
        cmap: string or matplotlib colormap
            Colormap for the figure
        alpha: float, default=0.25
            Opacity for the background
        figsize: tuple, default=(8, 6)
            Figure size for the figure
        ax: matplotlib axis, default=None
            If supplied, plot on axis given
    """
    #cmap = 'RdBu'
    
    l, r, d, u = bounds
    
    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    ymin, ymax = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Check if binary or categorical classification
    if model.layers[-1].units == 1:
        labels = (labels > 0.5).astype(int)
    else:
        labels = np.argmax(labels, axis=1)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, z, cmap=cmap, alpha=alpha)
    
    # Good for when we want to plot adversarial examples after
    if train_only:
        ax.scatter(*x_train.T, c=y_train, cmap=cmap, edgecolor='k')
    else:
        ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolor='k')
    ax.set_xlim(l, r)
    ax.set_ylim(d, u)


#########
# Utils #
#########

def decode_preds(preds, k=3):
    """
    Get top k predictions. Code borrowed from
    keras.applications.decode_predictions
    
    # Arguments:
        preds: predictions from model.predict()
        k: int, default=3
            Number of top predictions
    
    # Returns:
        decoded_preds: list
            The predictions sorted with label names
    
    # Raises
        ValueError:
            If k is greater than number of classes
    """
    if k > preds.shape[1]:
        raise ValueError("k must be less than or equal "
                         "to the number of classes")
    if preds.shape[1] == 10:
        label_names = CIFAR10_LABEL_NAMES
    else:
        label_names = OID_3CLASS_LABEL_NAMES
    
    decoded_preds = []
    for pred in preds:
        top_indices = pred.argsort()[-k:][::-1]
        result = [(label_names[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        decoded_preds.append(result)

    return decoded_preds


def print_norms(img, adv):
    """
    Prints L0, L1, L2, and Linf distance between img and adv

    # Arguments
        img: np.array
            Clean image
        adv: np.array
            Adversarial Example
    
    # Raises
        ValueError: if img and adv shapes don't match
    """
    if img.shape != adv.shape:
        raise ValueError("img and adv shapes don't match")
    diff = (adv - img).flatten()
    print("L0 norm:", np.linalg.norm(diff, ord=0))
    print("L1 norm:", np.linalg.norm(diff, ord=1))
    print("L2 norm:", np.linalg.norm(diff))
    print("Linf norm:", np.linalg.norm(diff, ord=np.inf))


def gradient_loss_wrt_input(x_test, y_test, model, normalization_trick=False):
    """
    Get gradient of loss with respect to the input
    
    # Arguments
        x_test: np.array
            Inputs to the model
        y_test: np.array
            must be one hot encoded
        model: keras model
        normalization_trick: Boolean, default=False
            If true, normalize the gradient with its L2 norm
    
    # Returns
        gradient: np.array
            The gradients of the loss wrt input
    """
    K.set_learning_phase(0)
    num_samples = x_test.shape[0]
    
    grads = K.gradients(model.total_loss, model.input)[0]
    if normalization_trick:
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    inputs = [model.input, model.sample_weights[0], model.targets[0]]
    get_gradients = K.function(inputs, [grads])
    gradient = get_gradients([x_test, np.ones(num_samples), y_test])[0]

    if num_samples == 1:
        gradient = np.squeeze(gradient)
    return gradient


def get_ortho_vector(x, seed=None):
    """
    Get random vector orthogonal to x with Gram-Schmidt method
    
    # Arguments
        x: np.array
            Starting vecot
        seed: int, default=None
            Seed for the random vector
    
    # Returns
        u: vector orthogonal to x
    """
    np.random.seed(seed)
    u = np.random.rand(x.shape[0])
    u -= (u.dot(x) * x) / np.linalg.norm(x)**2
    # Normalize
    u = u / np.linalg.norm(u)
    return u


def process_df(df, dataset):
    """
    Removes loss and fixes csv by doing transpose. Not really meant
    to be used in general.

    # Arguments
        df: pd.DataFrame
            Dataframe to process
        dataset: string
            'cifar10' or 'oid'

    # Returns
        df: pd.DataFrame
            The processed DataFrame
    """
    if dataset == 'cifar10':
        split_point = 4
        df.drop(columns=df.columns[[3, 5, 7, 9]], inplace=True)
    else:
        split_point = 3
        df.drop(columns=df.columns[[3, 5, 7]], inplace=True)
    df.columns = ['L1', 'L2', 'L_inf'] + [name + '_Acc' for name in df.index.tolist()]
    
    # This exists because I made a mistake on my csv script
    tmp = df.iloc[:, -split_point:]
    tmp = pd.DataFrame(tmp.values.T, index=tmp.index, columns=tmp.columns)
    df = pd.concat([df.iloc[:, :split_point], tmp], axis=1)
    
    return df


############
# Plotting #
############

def decision_regions(img, model, img_label=None, gradient=None, ortho=None,
                     sign_method=True, bounds=(-5, 5), num=100, cmap='jet',
                     plot_origin=False, grad_norm_trick=False, 
                     title='Decision Regions', xlab='Gradient Direction',
                     ylab='Orthogonal Direction', figsize=(8, 6), img_ref=False,
                     countplot=False, seed=None):
    """
    Plot a filled contour plot of the decision regions of the model.
    The xaxis is the direction of the gradient of the loss with respect
    to the input image, and the yaxis is a random direction orthogonal
    to the gradient. If the gradent and ortho is not given, then the
    function automatically makes them from the model and the seed.
    Optionally, you can supply your own vectors (does not need to be
    the gradient and ortho), and the vector supplied for 'gradient'
    will be the xaxis and the 'ortho' vector will be the yaxis.
    
    Idea from the following paper: "Delving into Transferable 
    Adversarial Examples and Black-box Attacks"
    https://arxiv.org/abs/1611.02770
    
    # Arguments
        img: np.array
            (w, h, 3) image to be perturbed
        model: keras model
        img_label: np.array, optional
            The true label for the given image. It should be one-hot
            encoded. Optional if you supply your own gradient and
            orthogonal vectors
        gradient: np.array, optional
            Direction for the x_axis. Should have the same dims as img
        ortho: np.array, optional
            Direction for the y_axis. Should have the same dims as img
        sign_method: Boolean, default True
            If true, direction of x_axis is the direction of the sign
            of the gradient. Else, x_axis direction is the direction 
            of the gradient.
        bounds: tuple, default=(-5, 5)
            Tuple of len 2 setting the range of the x and y axis
            perturbation.
        num: int, default=100
            Number of points between lower bound and upper bound
        cmap: str or matplotlib.cm, default='jet'
            Colormap to use for the contour plot
        plot_origin: Boolean, default=False
            If True, plot the origin in black
        grad_norm_trick: Boolean, default=False
            If True, normalize the gradient when computing it with
            gradient_loss_wrt_input()
        title: str
        xlab: str
        ylab: str
        figsize: tuple, default=(8, 6)
            Figure size for decision region plot
        img_ref: Boolean, default=False
            If True, show the 'most perturbed' image. Used as a
            visual reference to see the magnitude of the perturbation
        countplot: Boolean, default=False
        seed: int, default=None
            Seed for the RNG when getting the vector orthogonal to
            the gradient.
        
    # Returns
        fig, ax
    
    # Raises
        ValueError: If gradient shape does not match ortho shape
    """
    if gradient is None or ortho is None:
        shape = img.shape
        if len(img_label.shape) == 1:
            img_label = np.expand_dims(img_label, axis=0)
        gradient = gradient_loss_wrt_input(np.expand_dims(img, axis=0),
                                           img_label, model, grad_norm_trick)
        # Test gradient not all zero
        assert gradient.any(), 'Zero gradient'
        
        if sign_method:
            gradient = np.sign(gradient)
        gradient = gradient.flatten()
        # Normalize gradient
        gradient = gradient / np.linalg.norm(gradient)
        ortho = get_ortho_vector(gradient, seed=seed)
        
        # Reshape gradient, ortho to shape of image
        gradient = gradient.reshape(shape)
        ortho = ortho.reshape(shape)
    elif gradient.shape != ortho.shape:
        raise ValueError('gradient shape ' + str(gradient.shape)
                         + ' does not match ortho shape' + str(ortho.shape))
    elif gradient.shape != img.shape:
        raise ValueError('gradient shape ' + str(gradient.shape)
                         + ' does not match img shape' + str(img.shape))
    
    # Make grid
    x = y = np.linspace(*bounds, num)
    xx, yy = np.meshgrid(x, y)
    z = np.c_[xx.flatten(), yy.flatten()]
    
    # Make grid of perturbations
    img_grid = np.concatenate([np.expand_dims(img, axis=0)] * z.shape[0])
    # Could be easily optimized probably
    for i in range(z.shape[0]):
        img_grid[i] += (z[i][0] * gradient) + (z[i][1] * ortho)
    img_grid = np.clip(img_grid, 0, 1)
    
    # Check most perturbed image
    if img_ref:
        plt.imshow(img_grid[0])
        plt.show()
    
    # Make predictions and reshape
    pred = model.predict(img_grid)
    num_classes = pred.shape[1] # Use this to infer dataset
    pred = np.argmax(pred, axis=1)
    pred = pred.reshape(xx.shape)
    
    # Get label names depending on dataset
    unique_labels = np.sort(np.unique(pred))
    if num_classes == 10:
        label_names = [CIFAR10_LABEL_NAMES[i] for i in unique_labels]
    else:
        label_names = [OID_3CLASS_LABEL_NAMES[i] for i in unique_labels]
    
    # Get levels [-0.5, 0.5, ...]
    levels = np.arange(len(unique_labels)) + 0.5
    levels = np.insert(levels, 0, -0.5)
    
    # Map labels to [0, num_unique]
    cont_pred = pred.copy()
    for i, x in enumerate(unique_labels):
        cont_pred[cont_pred == x] = i
    
    # Plot filled contour
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if plot_origin:
        ax.plot([0], [0], marker='o', markersize=4, color="black")
    CS = ax.contourf(xx, yy, cont_pred, levels=levels, cmap=cmap)
    # Set title and axis
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    # Set colorbar stuff
    cbar = fig.colorbar(CS)
    cbar.set_clim(-0.5, len(unique_labels) - 0.5)
    cbar.set_ticks(list(range(len(unique_labels))))
    cbar.set_ticklabels(label_names)
    cbar.ax.set_ylabel('Class Label')
    plt.show()
    
    if countplot:
        pred = pred.flatten()
        print(Counter(pred))
        sns.countplot(pred)
        plt.show()
    
    return fig, ax


def grad_cam(model, img, label, conv_layer_idx, process=True):
    """
    Grad-CAM: Gradient based class activation map
    
    Code taken from Chollet's Deep Learning for Python book
    https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
    
    # Arguments
        model: keras model
        img: np.array
            image
        label: int
            Target (not one-hot encoded) to visualize
        conv_layer_idx: int
            Layer index for the last convolutional layer
        process: Boolean, default=True
            If true, make heatmap look nice

    # Returns
        heatmap: np.array
            Grad-CAM heatmap
    # Raises
        ValueError:
            If image shape is not len 3 or 4
    """
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    elif len(img.shape) != 4:
        raise ValueError('img not shape (1, w, h, 3)')
    
    class_output = model.output[:, label]
    last_conv_layer = model.layers[conv_layer_idx]
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    #print('Pooled grads:', pooled_grads.shape[0])
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    for i in range(pooled_grads.shape[0]): # 512
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    if process:
        #print(heatmap.max())
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / heatmap.max()
        heatmap = cv2.resize(heatmap, (32, 32))
    
    return heatmap


def plot_pair(img, adv, model=None, suptitle=None):
    """
    Visualize clean image, purturbation, and adversarial example
    
    # Arguments
        img: np.array
            The clean image with shape (w, h, 3)
        adv: np.array
            The adversarial example with shape (w, h, 3)
        model: keras model
        suptitle: string, default=None
    
    # Returns
        fig, axs:
    
    # Raises
        ValueError: If image or adv shape does not have len 3
    """
    if len(img.shape) != 3 or len(adv.shape) != 3:
        raise ValueError("Image shape must have len 3")
    
    # Get perturbation
    diff = img - adv
    diff = diff + abs(diff.min())
    diff = diff / diff.max()       
    
    # Make subplots and show three images
    fig, axs = plt.subplots(1, 3, figsize=(10, 9))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[1].imshow(diff)
    axs[1].axis('off')
    axs[2].imshow(adv)
    axs[2].axis('off')
    
    if model:
        # Get probabilities
        img_pred = model.predict(np.expand_dims(img, axis=0))
        adv_pred = model.predict(np.expand_dims(adv, axis=0))
        diff_pred = model.predict(np.expand_dims(diff, axis=0))
        
        num_classes = img_pred.shape[1]

        # Get confidence and index for each image
        img_pred_conf = np.max(img_pred)
        adv_pred_conf = np.max(adv_pred)
        diff_pred_conf = np.max(diff_pred)
        img_pred_label = np.argmax(img_pred)
        adv_pred_label = np.argmax(adv_pred)
        diff_pred_label = np.argmax(diff_pred)
        
        # Get label depending on dataset
        if num_classes == 10:
            img_pred_label = CIFAR10_LABEL_NAMES[img_pred_label]
            adv_pred_label = CIFAR10_LABEL_NAMES[adv_pred_label]
            diff_pred_label = CIFAR10_LABEL_NAMES[diff_pred_label]
        else:
            img_pred_label = OID_3CLASS_LABEL_NAMES[img_pred_label]
            adv_pred_label = OID_3CLASS_LABEL_NAMES[adv_pred_label]
            diff_pred_label = OID_3CLASS_LABEL_NAMES[diff_pred_label]
        
        # Set titles
        axs[0].set_title("Original Image" + "\nPred: " + str(img_pred_label)
                         + ' (' + str(round(img_pred_conf * 100, 1)) + '%)')
        axs[1].set_title("Difference" + "\nPred: " + str(diff_pred_label)
                         + ' (' + str(round(diff_pred_conf * 100, 1)) + '%)')
        axs[2].set_title("Adversarial Image" + "\nPred: " + str(adv_pred_label)
                         + ' (' + str(round(adv_pred_conf * 100, 1)) + '%)')
        fig.suptitle(suptitle, x=0.51, y=0.72, fontsize=20)
    else:
        axs[0].set_title("Original Image")
        axs[1].set_title("Difference")
        axs[2].set_title("Adversarial Image")
        fig.suptitle(suptitle, x=0.51, y=0.69, fontsize=20)
    
    return fig, axs


##############
# Never used #
##############

def gradient_output_wrt_input(model, img, normalization_trick=False):
    """
    Get gradient of softmax with respect to the input.
    Must check if correct.
    
    Do not use
    
    # Arguments
        model:
        img:
    
    # Returns
        gradient:
    """    
    grads = K.gradients(model.output, model.input)[0]
    if normalization_trick:
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [grads])
    grad_vals = iterate([img])[0]

    gradient = grad_vals[0]
    return gradient

def confuse_matrix(y_test, y_pred, label_names='cifar10'):
    """
    Plot confusion matrix
    
    # Arguments
        y_test: True labels
        y_pred: Predicted labels
        label_names: 'cifar10' or 'oid_3class'
    """
    confuse = confusion_matrix(y_test, y_pred)
    print(confuse)
    
    if label_names == 'cifar10':
        label_names = list(CIFAR10_LABEL_NAMES.values())
    else:
        label_names = list(OID_3CLASS_LABEL_NAMES.values())
    
    fig = ff.create_annotated_heatmap(confuse, x=label_names, y=label_names, 
                                      colorscale='Viridis')
    layout = go.Layout(title = "Confusion Matrix", titlefont=dict(size=32),
                       xaxis={'title': 'Predicted Labels', 'side': 'bottom',
                              'titlefont': dict(size=20), 
                              'tickfont': dict(size=14)}, 
                       yaxis={'title': 'True Labels', 
                              'titlefont': dict(size=20), 
                              'tickfont': dict(size=14)})
    fig.layout.update(layout)
    py.iplot(fig)
