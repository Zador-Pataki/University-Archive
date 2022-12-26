import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import argparse
from autofeat import FeatureSelector
from fancyimpute import SimpleFill, KNN, SoftImpute, IterativeImputer, IterativeSVD, MatrixFactorization, NuclearNormMinimization, BiScaler

base= '/cluster/work/riner/patakiz/AML/'
x_test = pd.read_csv(base+"X_test.csv")
x = pd.read_csv(base+"X_train.csv")
y = pd.read_csv(base+"y_train.csv")

x_test.drop(columns=["id"], inplace=True)
y.drop(columns=["id"], inplace=True)
x.drop(columns=["id"], inplace=True)
# x = x.to_numpy()
# x = y.to_numpy()

parser = argparse.ArgumentParser()
parser.add_argument("--imp", help='model', type=str, required=True)
args = parser.parse_args()
results=[]

#('simple', 'KNN', 'soft', 'iter', 'svd',
#'matrix', 'nuclear', 'bi')
n_neighbors = [3, 4, 5, 6, 7, 8]
homogenous_columns = []
for column_name in x.columns:
    if x[column_name].nunique() == 1:
        homogenous_columns.append(column_name)
x.drop(homogenous_columns, inplace=True, axis=1)

model = CatBoostRegressor(n_estimators = 9800, verbose=0)
f = FeatureSelector(verbose=0)
if 'simplemean' == args.imp:
    x = SimpleFill('mean').fit_transform(x)
    x = f.fit_transform(x,y)

    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))
if 'simplemedian' == args.imp:
    x = SimpleFill('median').fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

# elif 'simplemean' == args.imp:
#     pipe = Pipeline(steps=[('impute', SimpleFill('median')),
#                         ('regressor', model)])
#     results.append(str(pipe))
#     score = cross_val_score(pipe, scoring='r2', refit=False, verbose=2, n_jobs=-1, cv=10)
#     results.append('mean score:', str(np.mean(score)))

elif 'knn' in args.imp:
    if '3' in args.imp:
        x = KNN(3).fit_transform(x)
    if '4' in args.imp:
        x = KNN(4).fit_transform(x)
    if '5' in args.imp:
        x = KNN(5).fit_transform(x)
    if '6' in args.imp:
        x = KNN(6).fit_transform(x)
    if '7' in args.imp:
        x = KNN(7).fit_transform(x)
    if '8' in args.imp:
        x = KNN(8).fit_transform(x)
    # x = f.fit_transform(x, y)

    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))


elif 'nuclear' in args.imp:
    x = NuclearNormMinimization().fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

elif 'soft' in args.imp:
    x = SoftImpute().fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

elif 'iter' in args.imp:
    x = IterativeImputer().fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

if 'svd' in args.imp:
    x = IterativeSVD().fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

if 'matrix' in args.imp:
    x = MatrixFactorization().fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

if 'bi' in args.imp:
    x = BiScaler().fit_transform(x)
    x = f.fit_transform(x, y)
    score = cross_val_score(model, x, y, scoring='r2', verbose=2, n_jobs=-1, cv=10)
    results.append('mean score:')
    results.append(str(np.mean(score)))

textfile = open(args.imp+"_autofeat.txt", "w")
for element in results:
    textfile.write(element + "\n")
textfile.close()
