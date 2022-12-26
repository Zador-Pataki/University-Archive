from sklearn.linear_model import ElasticNetCV, RidgeCV, PassiveAggressiveRegressor, ARDRegression
from sklearn.linear_model import HuberRegressor
import argparse, pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
#parser.add_argument("--data", type=str, required=True)
parser.add_argument("--numlgbm", type=int, required=False, nargs='+')
parser.add_argument("--numcat", type=int, required=False, nargs='+')
parser.add_argument("--numet", type=int, required=False, nargs='+')
parser.add_argument("--numxgb", type=int, required=False, nargs='+')
parser.add_argument("--numsvr", type=int, required=False, nargs='+')

args = parser.parse_args()
models_list = []
cv_targets_list = []
"""
if args.data == 'Kacper1-1':
    x_train = pd.read_csv("/cluster/work/riner/patakiz/AML/x_train_"+args.data+".csv").to_numpy()
    x_test = pd.read_csv("/cluster/work/riner/patakiz/AML/x_test_"+args.data+".csv").to_numpy()
    y = pd.read_csv("/cluster/work/riner/patakiz/AML/y_train_"+args.data+".csv").to_numpy()[:,-1]
else:
    x_train = pd.read_csv("/cluster/work/riner/patakiz/AML/x_train_"+args.data+".csv").to_numpy()
    x_test = pd.read_csv("/cluster/work/riner/patakiz/AML/x_test_"+args.data+".csv", header=None).to_numpy()
    y = pd.read_csv("/cluster/work/riner/patakiz/AML/y_train_"+args.data+".csv").to_numpy()[:,-1]
"""
y1 = pd.read_csv("/cluster/work/riner/patakiz/AML/smart_stacking/testing1_1_1_1_.csv").to_numpy()[:,-1]
y2 = pd.read_csv("/cluster/work/riner/patakiz/AML/smart_stacking/1_1_1.csv").to_numpy()[:,-1]
y3 = pd.read_csv("/cluster/work/riner/patakiz/AML/smart_stacking/best.csv").to_numpy()[:,-1]

a,b, c = 1,1, 4
y = (a*y1+b*y2+c*y3)/(a+b+c)

out_df = pd.DataFrame(y, columns = ["y"])
out_df['id'] = out_df.index
out_df=out_df[["id","y"]]
out_df.to_csv('two_mine_one_best'+str(a)+'_'+str(b)+'_'+str(c)+'_'+'.csv',sep=',',index=False)

"""
if args.numsvr:
    for i in args.numsvr:
        try:
            model, y_cv = pickle.load(open('svr/'+args.data+'/'+str(i)+'.pkl', 'rb'))
            models_list.append(model)
            cv_targets_list.append(y_cv[:, np.newaxis])
        except:pass
        
if args.numxgb:
    
    for i in args.numxgb:
        try:
            model, y_cv = pickle.load(open('xgb/'+args.data+'/'+str(i)+'.pkl', 'rb'))
            models_list.append(model)
            cv_targets_list.append(y_cv[:, np.newaxis])
        except:pass
        
if args.numlgbm:
    for i in args.numlgbm:
        try:
            model, y_cv = pickle.load(open('lgbm/'+args.data+'/'+str(i)+'.pkl', 'rb'))
            models_list.append(model)
            cv_targets_list.append(y_cv[:, np.newaxis])
        except:pass

if args.numcat:
    for i in args.numcat:
        try:
            model, y_cv = pickle.load(open('cat/'+args.data+'/'+str(i)+'.pkl', 'rb'))
            models_list.append(model)
            cv_targets_list.append(y_cv[:, np.newaxis])
        except:pass
if args.numet:
    for i in args.numet:
        try:
            model, y_cv = pickle.load(open('et/'+args.data+'/'+str(i)+'.pkl', 'rb'))
            models_list.append(model)
            cv_targets_list.append(y_cv[:, np.newaxis])
        except:pass

cv_targets = np.concatenate(cv_targets_list, axis=1)

base_model = ElasticNetCV(n_alphas=5000, eps=0.0005, cv=10, n_jobs=-1)
# base_model = PassiveAggressiveRegressor()
# base_model = RidgeCV()
# base_model = HuberRegressor()
#base_model = ARDRegression()
base_model.fit(cv_targets, y)

meta_x_list = []

for model in models_list:
    meta_x_list.append(model.predict(x_test)[:,np.newaxis])
meta_x = np.concatenate(meta_x_list, axis=1)


def predict_based_on_model(model, x_test):
    y_pred = model.predict(x_test)
    out_df = pd.DataFrame(y_pred, columns = ["y"])
    out_df['id'] = out_df.index
    out_df=out_df[["id","y"]]
    out_df.to_csv(args.data+'_'+'_cat_'+str(args.numcat)+'_et_'+str(args.numet)+'_lgbm_'+str(args.numlgbm)+'_xgb_'+str(args.numxgb)+'_svr_'+str(args.numsvr)+'.csv',sep=',',index=False)

predict_based_on_model(model=base_model,
                       x_test=meta_x)"""
