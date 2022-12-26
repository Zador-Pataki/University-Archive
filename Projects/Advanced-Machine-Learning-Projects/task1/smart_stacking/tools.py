from sklearn.model_selection import KFold, cross_val_predict
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
def get_KFold_model_data(model, X, y, NN=False):
    models_list = []
    cv_targets_list = []
    y_tests = []
    X_tests = []
    kf = KFold(n_splits=10, shuffle=False)

    out = Parallel(n_jobs=-1, verbose=100)(
        delayed(do_task)(i, indices, model, X,y, NN) for i, indices in enumerate(kf.split(X)))
    return out

def do_task(i, indices, model, X, y, NN):
    train_index, test_index = indices
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    cv_target = cross_val_predict(model, X_train, y_train, cv=10, n_jobs=-1)
    return dict(model=model, cv_target=cv_target, index=i, y_test=y_test, y_train=y_train, x_test=X_test)

