from preprocess import Data
# from PIL import Image
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
seed = 9
'''
@author Sam
'''


# this function loads the data, shouldn't be needed after putting things in App.py
def load_data():
    '''
    Load the data for devloping the svm model.

    Returns
    -------
    X_train : Numpy array
        Array of data to train on.
    X_test : Numpy array
        Labels for training data.
    y_train : Numpy array
        Array of data to test.
    y_test : Numpy array
        Labels for test data.
    '''
    print('Setting up Data')
    data = Data(scale=0.15)
    X_train, X_test, y_train, y_test = data.splitData(random_state=seed)
    X_train = data.extra_processing(X_train, grayscale=True, flatten=True)
    X_test = data.extra_processing(X_test, grayscale=True, flatten=True)
    return X_train, X_test, y_train, y_test


def train_svm(X, y):
    '''
    Trains the svm on labelled training data.

    Parameters
    ----------
    X : The training data.

    y : The labels for the training data.

    Returns
    -------
    svm : The trained svm model.
    '''
    print("Creating SVM")
    seed = 9
    svm = SVC(kernel='poly', random_state=seed, verbose=1, max_iter=2500)
    print("Training")
    svm.fit(X, y)
    return svm


def evaluate_svm(X, y, model):
    '''
    Evaluates a trained svm given a labelled testing set.

    Parameters
    ----------
    X : The training data.

    y : The labels for the training data.

    model : The trained model to evaluate.

    Returns
    -------
    acc : The model's accuracy.

    f1_macro : The model's macroaveraged F1 score.
    '''
    acc = model.score(X, y)
    pred = model.predict(X)
    f1_macro = f1_score(y, pred, average="macro")
    return acc, f1_macro


def finetune_svm(X, y, model):
    '''
    Uses a model type to test out a grid of hyperparameters, and hopefully find the best settings.

    Parameters
    ----------
    X : The training data.

    y : The labels for the training data.

    model : The trained model to evaluate.

    Returns
    -------
    best_model : The most performant model.
    '''
    gridsearch = GridSearchCV(estimator=model,
                              param_grid={'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                          'C': [1, 10, 100, 1000],
                                          'tol': [1e-4],
                                          'max_iter': [100],
                                          'random_state': [0]},
                              verbose=2,
                              scoring='f1_macro')
    best_model = gridsearch.fit(X, y)
    print(gridsearch.best_params)
    return best_model


X_train, X_test, y_train, y_test = load_data()
# svm = train_svm(X_train, y_train)
svm = finetune_svm(X_train, y_train, SVC())
acc, f1 = evaluate_svm(X_test, y_test, svm)
print('Accuracy: {}, Macroaveraged F1 score: {}'.format(acc, f1))
