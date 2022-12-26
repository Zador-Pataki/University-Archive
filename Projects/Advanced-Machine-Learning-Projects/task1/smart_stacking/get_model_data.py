
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)

parser.add_argument("--num", type=int, required=True)
args = parser.parse_args()

if args.data == 'Kacper1-1':
    models = [XGBRegressor(booster = 'dart', gamma=0.2, n_estimators = 2000, reg_alpha = 0, reg_lambda = 1.5),
               XGBRegressor(booster = 'dart', gamma=0, n_estimators = 1600, reg_alpha = 0.5, reg_lambda = 0.5),
               XGBRegressor(booster = 'dart', gamma=0.2, n_estimators = 1300, reg_alpha = 0.5, reg_lambda = 0.5),
               XGBRegressor(booster = 'dart', gamma=0, n_estimators = 1000, reg_alpha = 0, reg_lambda = 1),
               ]
else:
     models = [XGBRegressor(booster = 'dart', gamma=0, n_estimators = 2000, reg_alpha = 0, reg_lambda = 0.5),
               XGBRegressor(booster = 'dart', gamma=0.2, n_estimators = 1600, reg_alpha = 0, reg_lambda = 0.5),
               XGBRegressor(booster = 'dart', gamma=0, n_estimators = 1300, reg_alpha = 0, reg_lambda = 1),
               XGBRegressor(booster = 'dart', gamma=0, n_estimators = 1000, reg_alpha = 0, reg_lambda = 1.5),
               ]

model = models[args.num]
print(model)

x_train = pd.read_csv("/cluster/work/riner/patakiz/AML/x_train_"+args.data+".csv").to_numpy()
x_test = pd.read_csv("/cluster/work/riner/patakiz/AML/x_test_"+args.data+".csv").to_numpy()
y = pd.read_csv("/cluster/work/riner/patakiz/AML/y_train_"+args.data+".csv").to_numpy()[:,-1]


results=[]
score = cross_val_score(model, x_train, y, verbose=2, cv=10, n_jobs=-1, scoring='r2')
results.append('scores:' + str(score))
results.append('mean score:' + str(np.mean(score)))



textfile = open(args.data+'/'+str(args.num) + ".txt", "w")
for element in results:
    textfile.write(element + "\n")
textfile.close()

model = models[args.num]
model.fit(x_train, y)

y_cv = cross_val_predict(model, x_train, y, cv=10, n_jobs=-1)

pickle.dump((model, y_cv), open(args.data+'/'+str(args.num)+'.pkl', 'wb'))
# model, y_cv = pickle.load(open(str(args.num)+'.pkl', 'rb'))
