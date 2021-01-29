'''
@author : Anish Lakkapragada
@date : 1 - 7 - 2021

This utils module is for all the "other stuff" used in ML. Here we provide a function to plot a confusion matrix,
a one-hot encoder, and a reverse one-hot encoder too.
'''

import numpy as np
import pandas as pd

def confusion_matrix(y_pred, y_test, plot = True) :
    """

    Confusion matrices are an often used tool for seeing how well your model did and where it can improve.

    A confusion matrix is just a matrix with the number of a certain class classified as another. So if your classifier
    predicted [0, 1] and the correct answers were [1, 0] - then you would get 1 zero predicted as a 1, 1 one predicted
    as a 0, and no 0s predicted as 0s and 1s predicted as 1s. By default this method will plot the confusion matrix, but
    if you don't want it just set it to False.

    Some warnings - make sure that y_pred and y_test are both 1D and start at 0. Also make sure that all predictions in
    y_pred exist in y_test. This will probably not be a big deal with traditional datasets for machine learning.

    To really understand this function try it out - it'll become clear with the visualization.

    :param y_pred: predictions (1D)
    :param y_test: labels (1D)
    :param plot: whether or not to plot, default True
    :return: the matrix, and show a visualization of the confusion matrix (if plot = True)
    """

    y_pred, y_test = np.array(y_pred, np.int).tolist(), np.array(y_test, np.int).tolist()

    if len(y_pred) != len(y_test) : raise ValueError("y_pred and y_test must have the same size.")

    possible_classes = set(y_test)
    if 0 not in possible_classes : raise ValueError("Labels must be start at 0.")

    confusion_matrix_dict = {lbl : {cls : 0 for cls in possible_classes} for lbl in possible_classes}

    for pred, label in zip(y_pred, y_test) :
        if pred not in y_test : raise ValueError("All predictions must belong to classes that are in y_test.")
        confusion_matrix_dict[label][pred] += 1

    conf_matrix = np.identity(len(confusion_matrix_dict)) #start the matrix off with an identity matrix
    for label, pred_dict in confusion_matrix_dict.items() :
        """label is the row, keys of pred_dict are the cols, values of the dict are the values"""
        for prediction, amount in pred_dict.items() :
            conf_matrix[label][prediction] = amount

    if plot :
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.clf()
        plt.cla()
        labels  = np.identity(len(possible_classes)).tolist()
        for row in range(len(labels)) :
            for col in range(len(labels[row])) :
                if row == col :
                    '''correct diagonal'''
                    labels[row][col] = "Correct : " + str(conf_matrix[row][col])
                else :
                    '''wrong diagonal'''
                    '''correct diagonal'''
                    labels[row][col] = "Incorrect : " + str(conf_matrix[row][col])
        labels = np.asarray(labels).reshape(len(possible_classes), len(possible_classes))
        sns.heatmap(conf_matrix, annot = labels, fmt = "", cmap = "Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Labels")
    return conf_matrix

def _change(zeros_and_index) :
    zeros, index = zeros_and_index
    #print("This is zeros : ", zeros)
    #print("This is index : ", index)
    zeros[index] = 1
    return zeros

def one_hot(indices, depth) :
    """
    So you've got a feature in a data where a certain value represents a certain category. For example it could be
    that 1 represents sunny, 2 represents rainy, and 0 represents cloudy. Well what's the problem? If you feed this
    to your model - it's going to think that 2 and 1 are similar, because they are just 1 apart - despite the fact
    that they are really just categories. To fix that you can feed in your features - say it's a list like [2, 2, 1,
    0, 1, 0] and it will be one hot encoded to whatever depth you please.

    Here's an example (it'll make it very clear) :
    >>> features_weather = [1, 2, 2, 2, 1, 0]
    >>> one_hot_features = one_hot(features_weather, depth  = 3) #depth is three here because you have 3 categories - rainy, sunny, cloudy
    >>> one_hot_features
    [[0, 1, 0], #1 is at first index
     [0, 0, 1], #2 is at second index
     [0, 0, 1], #2 is at second index
     [0, 0, 1], #2 is at second index
     [0, 1, 0], #1 is at first index
     [1, 0, 0]] #0 is at 0 index

    That looks like our data features, one hot encoded at whatever index. Make sure to set the depth param correctly.

    For these such things, play around - it will help.

    :param indices: features_weather or something similar as shown above
    :param depth: How many categories you have
    :return: one-hotted features
    """


    indices = np.array(indices, dtype = np.int).flatten()
    zeros = np.zeros((len(indices), depth))
    df = pd.DataFrame({'col1' : zeros.tolist(), 'col2' : indices.tolist()})
    return np.apply_along_axis(_change, 1, df)

def revert_one_hot(one_hot_data) :
    '''
    Say from the one_hot() data you've gotten something like this :

    [[0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]]

    and you want to change it back to its original form. Use this function - it will turn that one-hotted data above to
    [2, 0, 1, 2, 0] - which is just the index of the one in each data point of the list. Hence it is reverting the
    one_hot transformation.

    :param one_hot_data: data in one_hot form. Must be a numpy array (2D)!
    :return: index of the one for each of the data points in a 1D numpy array
    '''

    if np.isnan(one_hot_data).any() : raise ValueError("No NaN values are allowed.")

    if not float(np.sum(one_hot_data)) == float(one_hot_data.shape[0]) :
        raise ValueError("Data isn't one-hotted encoded properly.")

    return np.apply_along_axis(np.argmax, 1, one_hot_data)

def revert_softmax(softmax_data) :
    '''
    Say from the softmax function (in neural networks) you've gotten something like this :

    [[0.2, 0.3, 0.5],
    [0.8, 0.1, 0.1],
    [0.15, 0.8, 0.05],
    [0.3, 0.3, 0.4],
    [0.7, 0.15, 0.15]]

    and you want to change it back to its pre-softmax form. Use this function - it will turn that softmax-ed data above to
    [2, 0, 1, 2, 0] - which is just the index of the one in each data point of the list. Hence it is reverting the
    softmax transformation.

    :param softmax_data: data in one_hot form. Must be a numpy array (2D)!
    :return: index of the one for each of the data points in a 1D numpy array
    '''

    if np.isnan(softmax_data).any(): raise ValueError("No NaN values are allowed.")

    return np.apply_along_axis(np.argmax, 1, softmax_data)
