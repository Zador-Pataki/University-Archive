from sklearn.model_selection import GridSearchCV, cross_val_score

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
import argparse
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument("--num", help='num', type=int, required=True)
parser.add_argument("--data", help='num', type=str, required=True)
args = parser.parse_args()
a, b, c, d = np.unravel_index(args.num, (3,2,3,2))
args = parser.parse_args()


x_train = pd.read_csv("/cluster/work/riner/patakiz/AML/x_train_"+args.data+".csv")
x_test = pd.read_csv("/cluster/work/riner/patakiz/AML/x_test_"+args.data+".csv")
y = pd.read_csv("/cluster/work/riner/patakiz/AML/y_train_"+args.data+".csv").to_numpy()[:, -1]

x = x_train.to_numpy()
x_test = x_test.to_numpy()

# x_test_df = np.load('xtest.npy')

results = []





#n_estim_list = list(np.arange(100+a*500, 12000, 1000))
n_estim_list = [1000]
learning_rate = [0.1]
booster = ['dart']
gamma_xgb = [[0, 0.2, 0.5][a]]
alphas = [[0, 0.5][b]]
lambdas = [[0.5, 1, 1.5][c]]
max_depth = [[4,5],[6,7]][d] 


results.append('learning_rate:' + str(learning_rate))
results.append('n_estimators: ' + str(n_estim_list))
results.append('gamma: ' + str(gamma_xgb))
results.append('reg_alpha: ' + str(alphas))
results.append('reg_lambda: ' + str(lambdas))
results.append('booster: ' + str(booster))
results.append('max_depth: ' + str(max_depth))

search = [(XGBRegressor(verbosity=0), {'reg_alpha': alphas, 'booster': booster, 'learning_rate':learning_rate, 'gamma': gamma_xgb, 'reg_lambda': lambdas, 'n_estimators':n_estim_list})]

for model, parameters in search:
    clf = GridSearchCV(model, parameters, scoring='r2', refit=False, verbose=2, n_jobs=-1,cv=10)
    clf.fit(x, y)
    results.append(str(model))
    results.append(str(clf.best_params_))
    results.append(str(clf.best_score_))
    print(results)

textfile = open(args.data+'/'+str(args.num)+".txt", "w")
for element in results:
    textfile.write(element + "\n")
textfile.close()
