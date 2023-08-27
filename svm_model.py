# CREATE AND TRAIN MODELS

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd


def train_svm_model(X_train, y_train):
    """
    :param X_train: feature train data
    :param y_train: label train data
    :return: best fit model from grid search
    """
    clf = svm.SVC(random_state=6, verbose=0, decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    cv = KFold(n_splits=3, random_state=3, shuffle=True)
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly', 'linear']}
    clf_grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=cv)
    grid_results = clf_grid.fit(X_train, y_train)
    best_fit_model = grid_results.best_estimator_
    print('Model training complete!')
    best_params = best_fit_model.get_params()
    print(f'\nParameters used:\n {best_params}')

    return best_fit_model


def svm_model_predict(model, X_test, y_test):
    """
    :param model: trained model to predict from
    :param X_test: feature test data
    :param y_test: label test data
    :return: classification report of predictions
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

