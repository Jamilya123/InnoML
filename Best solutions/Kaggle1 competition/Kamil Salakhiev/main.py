#!/Users/kamil/miniconda3/envs/venv/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import *;
from sklearn.feature_extraction import DictVectorizer as DV
from common.feature_transformations import *
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import *
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import RFECV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.ensemble import VotingClassifier

from pprint import pprint
from time import time

import itertools

def get_features(df,  encoder=None):
    # ['workclass','education','marital-status','occupation', 'relationship', 'race', 'sex', 'native-country']
    # ['age','capital-gain', 'capital-loss', 'hours-per-week']
    '''
    score:0.862
    pca__n_components: 40
	svm__gamma: 0.1
    feature_columns_categorical = ['sex', 'education', 'native-country', 'marital-status', 'occupation']
    feature_columns_numerical = ['capital-gain', 'capital-loss']
    '''

    '''
    Best score: 0.862
    Best parameters set:
    	pca__n_components: 2
    cat = ['sex', 'education', 'native-country', 'marital-status', 'relationship', 'occupation']
    num = ['capital-gain', 'capital-loss']
    '''
    '''
    Best score: 0.863
    Best parameters set:
    	pca__n_components: 3
    	rand__max_depth: 11
    	rand__n_estimators: 15
    cat = ['sex', 'education', 'native-country', 'marital-status', 'relationship', 'occupation']
    num = ['capital-gain', 'capital-loss']
    '''

    '''
    Best score: 0.860
    Best parameters set:
    	knn__n_neighbors: 13
    	knn__p: 1
    	knn__weights: 'uniform'
    	pca__n_components: 20
    cat = ['sex', 'education', 'native-country', 'marital-status', 'relationship', 'occupation']
    num = ['capital-gain', 'capital-loss']
    '''
    '''
    Best score: 0.865
    Best parameters set:
    	ada__algorithm: 'SAMME.R'
    	ada__base_estimator: DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')
    	ada__n_estimators: 900
        cat = ['sex', 'education', 'native-country', 'marital-status', 'relationship', 'occupation']
        num = ['capital-gain', 'capital-loss']
    '''
    '''
    Best score: 0.863
    Best parameters set:
            bag__base_estimator: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=1,
               weights='uniform')
            bag__max_features: 0.7
            bag__max_samples: 0.7
    '''
    '''
    Best score: 0.870
    Best parameters set:
        ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),algorithm='SAMME.R',n_estimators=900)
        rfecv = RFECV(estimator=ada, step=1, scoring='accuracy',verbose=1)
    '''
    # cols = df.columns.values
    # vals = df.values
    # imp = Imputer(missing_values='?', strategy='mean', axis=0)
    # vals = imp.fit_transform(vals)
    # df = pd.Dataframe(vals,columns=cols)
    cat=['workclass','education','marital-status','occupation', 'relationship', 'race', 'sex', 'native-country']
    num=['age','capital-gain', 'capital-loss', 'hours-per-week']
    if (len(cat) == 0 or len(num) == 0):
        return 'con'
    if encoder is None:
        Xseries_categorical = [get_one_hot_encoding(df[x]) for x in cat]
        # Xseries_categorical = [df[x] for x in feature_columns_categorical]
    else:
        Xseries_categorical = [get_one_hot_encoding_with(df[x],encoder) for x in cat]
    Xseries_numerical = [df[x] for x in num]
    X = pd.concat(Xseries_categorical+Xseries_numerical, axis=1)
    return X.values

def get_categories(df):
    return df['Category'].values

def create_submission_with_model(model):
    test = pd.read_csv("mytest.csv", sep=",", header = 0)
    XTest = get_features(test)
    ypred = model.predict(XTest)
    index = np.arange(len(ypred))
    res = np.array(list(map(lambda x : [np.int(x[0]),np.int(x[1])],zip(index,ypred))))
    f = open('submission.csv', 'w')
    f.write('Id,Category\n')
    for x in res:
        f.write(str(int(x[0])) + ',' + str(int(x[1])) + '\n')
    f.close()

def plot(X, y):
    pca = PCA(n_components=5)
    pca.fit(X)
    X_reduced = pca.transform(X)
    import pylab as pl
    pl.scatter(X_reduced[:,3], X_reduced[:,4], c=y, cmap="RdYlBu", alpha=0.1)
    plt.show()
    exit()

def plot_models_hist(train):
    cats = ['workclass','education','marital-status','occupation', 'relationship', 'race', 'sex', 'native-country']
    # ['age','capital-gain', 'capital-loss', 'hours-per-week']
    zeros = train[train.Category==0]
    ones = train[train.Category==1]
    zeros2total=len(zeros)/len(train)
    ones2total=len(ones)/len(train)
    for cat in cats:
        my_dpi=96
        plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
        workUniq = np.unique(train[cat].values)
        # print(workUniq)
        data = []
        data.append([len(zeros[zeros[cat]==x])/zeros2total for x in workUniq])
        data.append([len(ones[ones[cat]==x])/ones2total for x in workUniq])
        # print(data)
        columns = workUniq
        rows = ['%d category' % x for x in [0,1]]
        max_col_sum = 0
        for i in range(len(workUniq)):
            Sum = np.sum(np.array(data)[:,i])
            if Sum > max_col_sum:
                max_col_sum = Sum
        values = np.arange(0, max_col_sum, max_col_sum/5)
        value_increment = 1
        # Get some pastel shades for the colors
        colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
        n_rows = len(data)
        index = np.arange(len(columns)) + 0.3
        bar_width = 0.4
        y_offset = np.array([0.0] * len(columns))
        cell_text = []
        for row in range(n_rows):
            plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
            y_offset = data[row]
            cell_text.append(['%1d' % (int(x)) for x in y_offset])
        # Reverse colors and text labels to display the last value at the top.
        colors = colors[::-1]
        cell_text.reverse()

        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              rowColours=colors,
                              colLabels=columns,
                              loc='bottom')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        plt.subplots_adjust(left=0.2, bottom=0.2)

        # plt.ylabel("Loss in ${0}'s".format(value_increment))
        plt.yticks(values * value_increment, ['%d' % val for val in values])
        plt.xticks([])
        plt.title(cat)
        # plt.show()
        plt.savefig(cat+'.png', dpi=my_dpi)
    exit()

def perform_search(X,y):
    # parameters = {
        # 'pca__n_components':[2,3,4,10,20,30,40,50],
        # 'knn__n_neighbors':(5,7,11,13,15),
        # 'knn__p':[1,2],
        # 'knn__weights':['uniform']
        # 'svm__gamma':np.logspace(-3, 3, 7)
        # 'rand__max_depth':[x for x in range(5,20,2)],
        # 'rand__n_estimators':[x for x in range(5,100,5)]
        # 'ada__base_estimator':[DecisionTreeClassifier(max_depth=1)],
        # 'ada__algorithm':['SAMME','SAMME.R'],
        # 'ada__n_estimators':[x for x in range(900,950,10)],
        # 'bag__base_estimator':[KNeighborsClassifier(n_neighbors=13),KNeighborsClassifier(n_neighbors=3),KNeighborsClassifier(n_neighbors=5),
            # KNeighborsClassifier(n_neighbors=13,p=1),KNeighborsClassifier(n_neighbors=3,p=1),KNeighborsClassifier(n_neighbors=5,p=1)],
        # 'bag__max_samples':[0.3,0.5,0.7],
        # 'bag__max_features':[0.3,0.5,0.7]
    # }

    # pipe = Pipeline([('pca', PCA()), ('knn',KNeighborsClassifier())])
    # pipe = Pipeline([('pca', PCA()), ('svm',svm.SVC(kernel='rbf'))])
    # pipe = Pipeline([('pca', PCA()), ('tree',DecisionTreeClassifier())])
    # pipe = Pipeline([ ('pca', PCA()),('rand', RandomForestClassifier())])
    # pipe = Pipeline([('ada',  AdaBoostClassifier())])
    # pipe= Pipeline([('bag',BaggingClassifier())])
    # grid_search = GridSearchCV(pipe, parameters, n_jobs=30, verbose=1)

    # '''
    # Voting:
    # '''
    # pipe1 = Pipeline([('pca', PCA()),('svm',svm.SVC(kernel='rbf',gamma=0.1,probability=True))])
    # pipe2 = Pipeline([('ada',  AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),algorithm='SAMME.R',n_estimators=900))])
    # pipe3 = Pipeline([('pca', PCA()),('rand', RandomForestClassifier(max_depth=12,n_estimators=13))])
    # pipe4 = Pipeline([('pca', PCA()),('knn',KNeighborsClassifier(p=1,n_neighbors=13))])
    # pipe5 = Pipeline([('bag',BaggingClassifier(KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #    metric_params=None, n_jobs=1, n_neighbors=5, p=1,
    #    weights='uniform'), max_features=0.7,max_samples=0.7))])
    #
    # # weights=[1,1,2,1]
    # # print(weights)
    # eclf = VotingClassifier(estimators=[('svm', pipe1), ('ada',pipe2), ('rand', pipe3), ('knn',pipe4)],voting='soft')
    #
    # # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=5)
    # eclf.fit(X,y)
    # print('cross validation started')
    # score = cross_val_score(eclf, X, y,n_jobs=-1,verbose=1).mean()
    # print(score)
    # exit()
    # return eclf
    # exit()
    # print(eclf.score(X_test, y_test))
    # exit()

    # '''
    # Bagging
    # '''
    # bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=13),max_samples=0.5, max_features=0.5)
    # print('cross validation started')
    # score = cross_val_score(bagging, X, y,n_jobs=-1,verbose=1).mean()
    # print(score)
    # exit()

    '''
    RFECV
    '''
    ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),algorithm='SAMME.R',n_estimators=900)
    rfecv = RFECV(estimator=ada, step=1, scoring='accuracy',verbose=1)
    print('fitting')
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(rfecv.grid_scores_)
    # score = cross_val_score(rfecv, X, y,n_jobs=-1,verbose=1).mean()
    # print(score)
    return rfecv
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipe.steps])
    # print("parameters:")
    # pprint(parameters)
    # t0 = time()
    # grid_search.fit(X, y)
    # print("done in %0.3fs" % (time() - t0))
    # print()
    #
    # print("Best score: %0.3f" % grid_search.best_score_)
    # print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # return grid_search

train = pd.read_csv("train.csv", sep=',', header = 0)
# plot_models_hist(train)

X = get_features(train)
y = get_categories(train)

print('Search for best classifier has started. Programm cat take a while to done all jobs')
model = perform_search(X,y)
# exit()
create_submission_with_model(model)
# plot(X,y)
exit()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=4)
# X_train, X_test = sklearn.preprocessing.normalize(X_train,norm='l2'), sklearn.preprocessing.normalize(X_test,norm='l2')
print("Data is preprocessed\n Learning Started")


# pipe = Pipeline([('pca', PCA(5)), ('svm',svm.SVC(kernel='poly'))]).fit(X_train,y_train)
# print(pipe.score(X_test,y_test))
knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
# print(knn.score(X_test, y_test))
# clf = svm.SVC(kernel='rbf', gamma=4).fit(X_train, y_train)
# print(clf.score(X_test, y_test))

# create_submission_with_model(knn)

exit()
