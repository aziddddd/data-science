#!/usr/bin/python

import math
import numpy as np
import pandas as pd
from sklearn import datasets, tree, metrics, model_selection, ensemble
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython.display import Image
from mpl_toolkits.mplot3d import Axes3D
import pydotplus
import itertools
import seaborn as sns

# preprocess country data
def gendata(features):

    # read data
    c = pd.read_csv('countries.csv', decimal=",")

    # set output and classes
    output = ['Birthrate']
    classes = ['low', 'medium', 'high']

    # shorten feature names
    c.columns = ['Country', 'Region', 'Population', 'Area', 'Density', 'Coastline', 'Migration', 'InfantMortality', \
                 'GDP', 'Literacy', 'Phones', 'Arable', 'Crops', 'OtherLand', 'Climate', 'Birthrate', 'Deathrate', \
                 'Agriculture', 'Industry', 'Service']

    # strip all whitespace from all columns
    c = c.applymap(lambda x: x.strip() if type(x) is str else x)

    # set index to country
    c.set_index('Country', inplace=True)

    # reduce to feature and type columns
    dataset = c[features + output]

    # drop duplicates and null values
    dataset = dataset.drop_duplicates().dropna()

    # make new birthrate class column
    btype = []
    for b in dataset.Birthrate:
        if (b < 15):
            btype.append('low')
        elif ((b >= 15) and (b < 30)):
            btype.append('medium')
        elif (b >= 30):
            btype.append('high')

    # remove original birth rate column
    dataset = dataset.drop(columns=['Birthrate'])
    # NOTE - alternative if using pandas < 0.20
    #del dataset['Birthrate']

    # append to dataset
    dataset['BRClass'] = pd.Series(btype, index=dataset.index)

    # return values
    return dataset.values


# clearer and simpler version of featureplot for Checkpoint 4
# keeping featureplot in for backwards compatibility
# restrict to 6 features
def cp4plot(data, target, features=None, classes=None):

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
    sub_y = math.ceil(len(pairs)/4.)
    full_y = sub_y * 3.5

    # set figure size
    plt.figure(1, figsize=(15, full_y))

    # enumerate over combinations
    for pairidx, pair in enumerate(pairs):

        # extract data for pair
        datapair = data[:, pair]

        # define new plot
        plt.subplot(sub_y, 4, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        # add labels
        plt.xlabel(features[pair[0]])
        plt.ylabel(features[pair[1]])

        # Plot the points
        for i, color in zip(range(n_classes), plt_colors):

            idx = np.where(target == classes[i])

            plt.scatter(datapair[idx, 0], datapair[idx, 1], c=color, label=classes[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")

    return

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

# plot decision tree using sklearn export_graphviz
# See: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# from command line "run dot -Tpng iris.dot -o tree.png"
# Inline logic adapted from https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
# note: requires pydotplus package
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

    return graph

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

# Note - expect convergence warning at small training sizes
def compare_traintest(data, target, params=None, split=0, scale='linear'):

    # define 0.01 - 0.1, 0.1 - 0.9, 0.91 - 0.99 sample if split array not defined
    if (split == 0):
        split = np.concatenate((np.linspace(0.01,0.09,9), np.linspace(0.1,0.9,9), np.linspace(0.91,0.99,9)), axis=None)

    print("parameters")
    print(params)

    print("Split sample:")
    print(split)

    train_scores = []
    test_scores = []

    for s in split:

        print("Running with test size of: %0.2f" % s)

        # get train/test for this split
        d = model_selection.train_test_split(data, target,
                                             test_size=s, random_state=0)

        # define classifer
        if params is not None:
            clf = tree.DecisionTreeClassifier(**params)
        else:
            clf = tree.DecisionTreeClassifier()

        # run classifer
        e, p = runML(clf, d)

        # get training and test scores for fit and prediction
        train_scores.append(clf.score(d[0], d[2]))
        test_scores.append(clf.score(d[1], d[3]))

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
    plt.ylim([0,1.01])
    plt.grid()
    plt.legend()

    return
