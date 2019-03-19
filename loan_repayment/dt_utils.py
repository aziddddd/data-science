import math
import pandas as pd
import numpy as np
import re
import pydotplus

import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from sklearn import model_selection, preprocessing, tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals.six import StringIO

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from tqdm import tqdm_notebook as tqdm

no_regex0   = re.compile(r'[\w\W]*Void')
no_regex1   = re.compile(r'[\w\W]*Charged Off')
no_regex2   = re.compile(r'[\w\W]*Collection')
no_regex3   = re.compile(r'Returned Item')
no_regex4   = re.compile(r'Withdrawn Application')
no_regex5   = re.compile(r'New Loan')
no_regex6   = re.compile(r'[\w\W]*Rescind')
no_regex7   = re.compile(r'[\w\W]*Bankruptcy')

yes_regex0  = re.compile(r'[\w\W]*Paid Off')

class loanError(Exception):
    """ An exception class for dt_utils """
    pass
    
def preprocess(payment_path, loan_path):
    label = {}
    label['badLoanBehaviour'] = np.array([0.0, 1.0])
    payment_processed = pd.read_csv(payment_path).sort_values(by=['loanId']).reset_index()
    raw_loans = pd.read_csv(loan_path).sort_values(by=['loanId']).reset_index()

    loans = raw_loans[['loanId', 'apr', 'loanAmount', 'originallyScheduledPaymentAmount', 'loanStatus', 'leadCost']].drop_duplicates(subset='loanId').dropna()

    df = pd.merge(loans, payment_processed, on='loanId')

    df = df.rename(columns = {'paymentStatus':'countPS'}).rename(columns = {'paymentStatus.1':'topPS'}).rename(columns = {'paymentStatus.2':'frequencyPS'})

    df['badLoanBehaviour'] = ''

    for idx, i in enumerate(df['loanId']):
        if no_regex0.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex1.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex2.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex3.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex4.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex5.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex6.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif no_regex7.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 1.0
            pass

        elif yes_regex0.match(df['loanStatus'][idx]):
            df.loc[idx:idx, 'badLoanBehaviour'].values[0] = 0.0
            pass

        else:
            raise loanError('Cannot determine the loan behaviour. [ loanId : {0} ]    [ loanStatus: {1} ]'.format(df['loanId'][idx], df['loanStatus'][idx]))

    df = df.drop(columns=['loanId']).drop(columns=['countPS']).drop(columns=['frequencyPS']).drop(columns=['index'])

    for feature in df.keys():
        if isinstance(df[feature].any(), str):
            label[feature] = df[feature].unique()
            df[feature] = df[feature].factorize()[0]
    
    return df, label

def featuresplot(data, target, label, features=None, classes=None):

    if (features is None):
        print("Please provide a list of feature names")
        return
    if (classes is None):
        print("Please provide a list of class names")
        return

    plt_colors = "rybgcm"
    n_classes = len(classes)

    if (len(features) > 6):
        print("Number of features is too high to plot")
        return

    # get pair list of permutations and get unique set
    n_features = data.shape[1]
    
    x = [sorted(i) for i in itertools.permutations(np.arange(n_features), r=2)]
    x.sort()

    pairs = list(k for k,_ in itertools.groupby(x))
    
    # set subplot layout
    sub_y = math.ceil(len(pairs)/3.)
    full_y = sub_y * 3.5

    # set figure size
    plt.figure(1, figsize=(15, full_y))

    # enumerate over combinations
    for pairidx, pair in enumerate(pairs):

        # extract data for pair
        datapair = data[:, pair]
        
        # define new plot
        plt.subplot(sub_y, 3, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        # add labels
        plt.xlabel(features[pair[0]])
        plt.ylabel(features[pair[1]])

        if features[pair[0]] == 'apr' or features[pair[0]] == 'loanAmount' or features[pair[0]] == 'originallyScheduledPaymentAmount' or features[pair[0]] == 'leadCost':
            pass
    
        elif features[pair[0]] == 'loanStatus' or features[pair[0]] == 'topPS':
            plt.xticks(np.linspace(0.0, len(label[features[pair[0]]]),len(label[features[pair[0]]])+1), label[features[pair[0]]], rotation='vertical', fontsize=8)    

        if features[pair[1]] == 'apr' or features[pair[1]] == 'loanAmount' or features[pair[1]] == 'originallyScheduledPaymentAmount' or features[pair[1]] == 'leadCost':
            pass

        elif features[pair[1]] == 'loanStatus' or features[pair[1]] == 'topPS':
            plt.yticks(np.linspace(0.0, len(label[features[pair[1]]]),len(label[features[pair[1]]])+1), label[features[pair[1]]], fontsize=8)
            pass    

        # Plot the points
        for i, color in zip(range(n_classes), plt_colors):

            idx = np.where(target == classes[i])

            plt.scatter(datapair[idx, 0], datapair[idx, 1], c=color, label=classes[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.legend(np.array(['Good Loan Behaviour', 'Bad Loan Behaviour']), loc='lower right', borderpad=0, handletextpad=0, bbox_to_anchor=(1.0, -0.5))
    plt.axis("tight")

    return

def evaluate_model_accuracy(model, data, target_label_1d, **params):
    print("k-Fold Cross Validation")
    print("Parameters")
    print(params)

    kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    cvscores = []
    for train, test in tqdm(kfold.split(data, target_label_1d)):
        # convert integers to dummy variables (i.e. one hot encoded)
        target_label = np_utils.to_categorical(target_label_1d).astype(float)

        # Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale both the training inputs and outputs
        data[train] = scaler.fit_transform(data[train])
        data[test] = scaler.fit_transform(data[test])

        # Fit the model
        model.fit(data[train], target_label[train], epochs=10, verbose=0)

        # evaluate the model
        scores = model.evaluate(data[test], target_label[test], verbose=0)
        print('{0:} : {1:0.2f}%'.format(model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print('Model Accuracy : {0:0.2f}% (+/- {1:0.2f}%)'.format(np.mean(cvscores), np.std(cvscores)))
    
    return np.mean(cvscores)

# Note - expect convergence warning at small training sizes
def compare_traintest(data, target, model, split=0, scale='linear', **params):
    #preprocess target data
    le = preprocessing.LabelEncoder()
    target_label = le.fit_transform(target)

    # convert integers to dummy variables (i.e. one hot encoded)
    target_label = np_utils.to_categorical(target_label).astype(float)
    
    # define 0.01 - 0.1, 0.1 - 0.9, 0.91 - 0.99 sample if split array not defined
    if len(split) == 0:
        split = np.concatenate((np.linspace(0.01,0.09,9), np.linspace(0.1,0.9,9), np.linspace(0.91,0.99,9)), axis=None)

    print("Parameters")
    print(params)
        
    print("Split sample:")
    print(split)

    train_scores = []
    test_scores = []

    for s in tqdm(split):
        print("Running with test size of: %0.2f" % s)

        # get train/test for this split
        d = model_selection.train_test_split(data, target_label,
                                             test_size=s, random_state=0)

        # get training and test data and targets
        train_data, test_data, train_target, test_target = d

        # Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale both the training inputs and outputs
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
        # Train the model
        model.fit(
            train_data,
            train_target,
            **params,
            validation_data=(test_data, test_target)
        )

        train_error_rate = model.evaluate(train_data, train_target, verbose=0)
        test_error_rate = model.evaluate(test_data, test_target, verbose=0)

        # get test scores for fit and prediction
        train_scores.append(train_error_rate[1])
        test_scores.append(test_error_rate[1])

    # plot results
    plt.figure(figsize=(15.0, 5.0))
    if (scale == 'log'):
        plt.yscale('log')
    else:
        plt.yscale('linear')
    plt.plot(split, train_scores, label='Training accuracy', marker='o')
    plt.plot(split, test_scores, label='Testing accuracy', marker='o')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Test sample proportion')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, 1.0, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim([min(split),max(split)])
    plt.ylim([0.7,1.01])
    plt.grid()
    plt.legend()

    return

# Rerun fit and prediction steps for later examples
def runML(clf, d):

    # get training and test data and targets
    train_data, test_data, train_target, test_target = d

    # fit classifier with data
    fit = clf.fit(train_data, train_target)

    # define expected and predicted
    expected = test_target
    predicted = clf.predict(test_data)

    # return results
    return [expected, predicted]

def plotDT(fit, feature_names, target_names, fname=None):
    if (fname is not None):
        tree.export_graphviz(fit, out_file=fname, filled=True, rounded=True,
                             special_characters=True,
                             feature_names=feature_names,
                             class_names=target_names)
        graph = 0

    else:
        dot_data = StringIO()
        tree.export_graphviz(fit, out_file=dot_data, filled=True, rounded=True,
                             special_characters=True,
                             feature_names=feature_names,
                             class_names=target_names)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.set_size('"15,15!"')

    return graph

# pretty print confusion matrix
# orginial: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_cm(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return

def featureImportance(clf, train_data, features):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print('Feature ranking:')

    for f in range(train_data.shape[1]):
        print('{0:d}.{1:>35}    {2:0.4f}'.format(f + 1, features[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title('Feature importances')
    plt.bar(range(train_data.shape[1]), importances[indices],
        color="r", yerr=std[indices], align='center')
    plt.xticks(range(train_data.shape[1]), [features[indices[f]] for f in range(train_data.shape[1])], rotation=30, horizontalalignment='right')
    plt.xlim([-1, train_data.shape[1]])
    plt.show()

# heat map for confusion matrices and parameter scans
# adapted from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def heatmap(d, labels=None, classes=None, title=None,
            palette="Green",
            normalize=False,
            annot=True):

    if normalize:
        d = d.astype('float') / d.sum(axis=1)[:, np.newaxis]

    ax = plt.subplot()

    # define colour map
    my_cmap = sns.light_palette(palette, as_cmap=True)

    # plot heatmap
    sns.heatmap(d, annot=True, ax=ax, cmap=my_cmap)

    # labels, title and ticks
    if (labels is not None):
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if (title is not None):
        ax.set_title('Confusion Matrix')
    if (classes is not None):
        ax.xaxis.set_ticklabels(classes[0])
        ax.yaxis.set_ticklabels(classes[1])

    return
