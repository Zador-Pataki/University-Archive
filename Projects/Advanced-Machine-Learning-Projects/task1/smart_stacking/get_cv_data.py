from xgboost import XGBRegressor

import pandas as pd
import argparse
import pickle
import sys
sys.path.append('/cluster/work/riner/patakiz/AML/smart_stacking')
from tools import get_KFold_model_data
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)

parser.add_argument("--num", type=int, required=True, choices=[0,1,2,3])
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



x_train = pd.read_csv("/cluster/work/riner/patakiz/AML/x_train_"+args.data+".csv").to_numpy()
x_test = pd.read_csv("/cluster/work/riner/patakiz/AML/x_test_"+args.data+".csv").to_numpy()
y = pd.read_csv("/cluster/work/riner/patakiz/AML/y_train_"+args.data+".csv").to_numpy()[:,-1]



model = models[args.num]

output = get_KFold_model_data(model, x_train, y)

models_list = [d['model'] for d in output]
cv_targets_list = [d['cv_target'] for d in output]
y_train = [d['y_train'] for d in output]
X_tests = [d['x_test'] for d in output]
y_tests = [d['y_test'] for d in output]
indices = [d['index'] for d in output]

print(indices)

pickle.dump(models_list, open(args.data+'/cv_data/'+str(args.num)+'_models_cv.pkl', 'wb'))
pickle.dump(X_tests, open(args.data+'/cv_data/'+str(args.num)+'_X_tests_cv.pkl', 'wb'))
pickle.dump(y_train, open(args.data+'/cv_data/'+str(args.num)+'_y_train_cv.pkl', 'wb'))
pickle.dump(y_tests, open(args.data+'/cv_data/'+str(args.num)+'_y_tests_cv.pkl', 'wb'))
pickle.dump(cv_targets_list, open(args.data+'/cv_data/'+str(args.num)+'_cv_targets.pkl', 'wb'))

