# Decision Tree module for ML Case Study
import numpy as np
import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
from main import load_data, data_processing, drop_date
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def get_X_and_y(df):
    y = np.array(df.pop('churn'))
    X = np.array(df)
    return X, y

def score(y_true, y_pred):
    f1 =  f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return f1, accuracy

def decision_tree(X, y, ):
    model = DecisionTreeClassifier()
    model = model.fit(X,y)
    return model

def print_tree(model):
    dot_data = export_graphviz(model, out_file='tree.dot')
    #graph = pydotplus.graph_from_dot_data(dot_data)
    #graph.write_pdf('DecisionTree.pdf')

def gd_model(X, y, subsample=.5, max_depth=10):
    return GradientBoostingClassifier(subsample=subsample, max_depth=max_depth).fit(X, y)

def rf_model(X, y, n_estimators=100):
    return RandomForestClassifier(n_estimators=n_estimators).fit(X, y)

def adb_model(X, y, n_estimators=50, learning_rate=0.1):
    adb = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
    return adb.fit(X, y)

def knn(X, y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    return knn.fit(X, y)

def plot_imp(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], yerr=[indices], # yerr: if not None, will be used to generate errorbar(s) on the bar chart
            color="r", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def gd_gridsearch():
    gradient_boosting_grid = {'learning_rate': [.1, .2, .3],
                              'max_depth': [3, 5, 7, 9],
                              'n_estimators': [100, 1000, 10000],
                              'subsample': [.3, .5, .7, .9],
                              'random_state': [1]}
    gb_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                 gradient_boosting_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='neg_mean_squared_error')
    gb_gridsearch.fit(X_train, y_train)
    gb_gridsearch.best_params_

def rf_gridsearch():
    random_forest_grid = {'max_depth': [3, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}
    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                 random_forest_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='neg_mean_squared_error')
    rf_gridsearch.fit(X_train, y_train)
    rf_gridsearch.best_params

if __name__ == '__main__':
    df = load_data()
    df = data_processing(df)
    df = drop_date(df)

    X, y = get_X_and_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    # first_tree = decision_tree(X_train, y_train)
    # dt_pred = first_tree.predict(X_test)
    # print 'DT F1: {}, accuracy: {}'.format(score(y_test, dt_pred)[0], score(y_test, dt_pred)[1])
    #
    # # GD
    # gd_pred = gd_model(X_train, y_train).predict(X_test)
    # print 'GD F1: {}, accuracy: {}'.format(score(y_test, gd_pred)[0], score(y_test, gd_pred)[1])
    #
    # # RF
    # rf_pred = rf_model(X_train, y_train).predict(X_test)
    # print 'RF F1: {}, accuracy: {}'.format(score(y_test, rf_pred)[0], score(y_test, rf_pred)[1])
    #
    # # adaboost
    # adb_pred = adb_model(X_train, y_train).predict(X_test)
    # print 'RF F1: {}, accuracy: {}'.format(score(y_test, adb_pred)[0], score(y_test, adb_pred)[1])
    #
    # # kNN
    # knn_pred = knn(X_train, y_train).predict(X_test)
    # print 'kNN F1: {}, accuracy: {}'.format(score(y_test, knn_pred)[0], score(y_test, knn_pred)[1])
