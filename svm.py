from preprocess import Data
# from PIL import Image
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np
seed = 9

'''
NOTES
data.X object contains a list of all the images

TODO
make svm finetuning loop over different initializations and over all hyperparameters
combine the code for unet and svm with App.py
'''


# this function loads the data, shouldn't be needed after putting things in App.py
def load_data():
    print('Setting up Data')
    data = Data(scale=0.15)  # TODO .55 for real training
    X_train, X_test, y_train, y_test = data.splitData(random_state=seed)
    X_train = data.extra_processing(X_train, grayscale=True, flatten=True)
    X_test = data.extra_processing(X_test, grayscale=True, flatten=True)
    return X_train, X_test, y_train, y_test


def train_svm(X, y):
    print("Creating SVM")
    seed = 9
    svm = SVC(kernel='poly', random_state=seed, verbose=1, max_iter=2500)
    print("Training")
    svm.fit(X, y)
    return svm


def evaluate_svm(X, y, model):
    acc = model.score(X, y)
    pred = model.predict(X)
    f1_macro = f1_score(y, pred, average="macro")
    return acc, f1_macro


def finetune_svm(X, y, model):
    # TODO maybe tune over gamma and degree as well (only matter for certain kernels)
    # print(model.get_params().keys())
    gridsearch = GridSearchCV(estimator=model,
                              param_grid={'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                          'C': [1, 10, 100, 1000],
                                          'tol': [1e-4],
                                          'max_iter': [100],# TODO cut this
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
