import numpy as np
import pandas as pd
import argparse
import pickle


from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from tqdm import trange
from sklearn.linear_model import HuberRegressor, OrthogonalMatchingPursuitCV, LarsCV, LassoLarsCV, ARDRegression, BayesianRidge
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--num", type=int, required=True)

"""
parser.add_argument("--numlgbm", type=int, required=False, nargs='+')
parser.add_argument("--numcat", type=int, required=False, nargs='+')
parser.add_argument("--numet", type=int, required=False, nargs='+')
parser.add_argument("--numnn", type=int, required=False, nargs='+')
"""
args = parser.parse_args()
if args.data == 'Kacper1-1':
    a,b = np.unravel_index(args.num, (11,5))
if args.data == 'Kacper2-1':
    a,b = np.unravel_index(args.num, (11,5))
if args.data == 'Kacper2-2':
    a,b = np.unravel_index(args.num, (8,5))
if args.data == 'Kacper2-3':
    a,b = np.unravel_index(args.num, (4,5))
if args.data == 'Kacper2-4':
    a,b = np.unravel_index(args.num, (7,5))


y_train = pickle.load(open('et/'+args.data+'/cv_data/'+str(0)+'_y_train_cv.pkl', 'rb'))
X_test = pickle.load(open('et/'+args.data+'/cv_data/'+str(0)+'_X_tests_cv.pkl', 'rb'))
y_tests = pickle.load(open('et/'+args.data+'/cv_data/'+str(0)+'_y_tests_cv.pkl', 'rb'))


# if args.numlgbm > 0:
#     for i in range(args.numlgbm):
#         model, y_cv = pickle.load(open('lgbm/'+str(i)+'.pkl', 'rb'))
#         models_list.append(model)
#         cv_targets_list.append(y_cv[:, np.newaxis])
#
# if args.numcat > 0:
#     for i in range(args.numcat):
#         model, y_cv = pickle.load(open('cat/'+str(i)+'.pkl', 'rb'))
#         models_list.append(model)
#         cv_targets_list.append(y_cv[:, np.newaxis])
#
if args.data == 'Kacper1-1':
    numcat = [[False,[6], [5], [4], [7], [6,5], [6,4], [6,7], [5,4], [5,7], [4,7]][a]]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numet = [False,[3], [4], [5], [6], [3,4], [3,5], [3,6], [4,5], [4,6], [5,6]]#, [3,4,5], [3,4,6], [3,5,6], [4,5,6], [3,4,5,6]] 
    numlgbm = [False,[4], [5], [6], [7], [4,5], [4,6], [4,7], [5,6], [5,7], [6,7]]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numsvr = [False,[0]]
    numxgb = [False,[1], [2], [1,2]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 

if args.data == 'Kacper2-1':
    numcat = [False,[0], [2], [3], [0,2], [0,3], [2,3]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
    #numcat = [False]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
    numet = [False,[0], [1], [2], [3], [0,1], [0,2], [1,2], [0,3],[1,3],[2,3]]#, [0,1,2], [0,1,2]]  
    numlgbm = [False,[0], [1], [2], [3], [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
    numsvr = [False,[0]]
    numxgb = [[False,[0], [1], [2], [0,1], [0,2], [1,2]][a]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
 
if args.data == 'Kacper2-2':
    numcat = [False]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numet = [False,[0], [1], [0,1]]#, [3,4,5], [3,4,6], [3,5,6], [4,5,6], [3,4,5,6]] 
    numlgbm = [False,[0], [1], [0,1]]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numsvr = [False,[0]]
    numxgb = [[False,[1], [2], [1,2]][a]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
 
if args.data == 'Kacper2-3':
    numcat = [False]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numet = [False,[0], [1], [0,1]]#, [3,4,5], [3,4,6], [3,5,6], [4,5,6], [3,4,5,6]] 
    numlgbm = [False,[0], [1], [0,1]]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numsvr = [False,[0]]
    numxgb = [[False,[0], [1], [0,1]][a]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
   
if args.data == 'Kacper2-4':
    numcat = [False]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numet = [False,[0], [1], [0,1]]#, [3,4,5], [3,4,6], [3,5,6], [4,5,6], [3,4,5,6]] 
    numlgbm = [False,[0], [1], [0,1]]#, [4,5,6], [4,5,7], [4,6,7], [5,6,7], [4,5,6,7]] 
    numsvr = [False,[0]]
    numxgb = [False,[0], [1], [2], [0,1], [0,2], [1,2]]#, [0,1,2], [0,1,2], [0,2,3], [1,2,3], [0,1,2,3]] 
print(numsvr)

svr_list =[]
cat_list =[]
et_list =[]
lgbm_list =[]
xgb_list =[]

svr_targets_list =[]
cat_targets_list =[]
et_targets_list =[]
lgbm_targets_list =[]
xgb_targets_list =[]

print('1')
if numsvr:
    for i in [0,1,2,3,4,5,6,7]:
        load = False
        for sig in numsvr:
            if sig:
                if i in sig:
                    load=True
        if load:
            svr_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
            svr_targets_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
        else:
            svr_list.append(None)
            svr_targets_list.append(None)
"""
if args.data == 'Kacper2-1':
    for i in [0]:
        svr_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
        svr_targets_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
if args.data == 'Kacper2-2':
    for i in [0]:
        svr_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
        svr_targets_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
if args.data == 'Kacper2-3':
    for i in [0]:
        svr_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
        svr_targets_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
if args.data == 'Kacper2-4':
    for i in [0]:
        svr_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
        svr_targets_list.append(pickle.load(open('svr/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
"""
#for i in [0,1,2,3,4,5,6,7]:
print('2')
if numcat:
    for i in [0,1,2,3,4,5,6,7]:
        load = False
        for sig in numcat:
            if sig:
                if i in sig:
                    load=True
        if load:
            cat_list.append(pickle.load(open('cat/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
            cat_targets_list.append(pickle.load(open('cat/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
        else:
            cat_list.append(None)
            cat_targets_list.append(None)
print('3')

if numet:
    for i in [0,1,2,3,4,5,6,7]:
        load = False
        for sig in numet:
            if sig:
                if i in sig:
                    load=True
        if load:
            et_list.append(pickle.load(open('et/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
            et_targets_list.append(pickle.load(open('et/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
        else:
            et_list.append(None)
            et_targets_list.append(None)
        
print('4')
if numlgbm:
    for i in [0,1,2,3,4,5,6,7]:
        load = False
        for sig in numlgbm:
            if sig:
                if i in sig:
                    load=True
        if load:
            lgbm_list.append(pickle.load(open('lgbm/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
            lgbm_targets_list.append(pickle.load(open('lgbm/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
        else:
            lgbm_list.append(None)
            lgbm_targets_list.append(None)
#for i in [0,1,2,3]:
print('5')
#for i in [0,1,2,3]:
if numxgb:
    for i in [0,1,2,3,4,5,6,7]:
        load=False
        for sig in numxgb:
            if sig:
                if i in sig:
                    load=True
        if load:
            xgb_list.append(pickle.load(open('xgb/'+args.data+'/cv_data/'+str(i)+'_models_cv.pkl', 'rb')))
            xgb_targets_list.append(pickle.load(open('xgb/'+args.data+'/cv_data/'+str(i)+'_cv_targets.pkl', 'rb')))
        else:
            xgb_list.append(None)
            xgb_targets_list.append(None)
print('6')
max_score = -100
top_model = []
if b = 2:
    assert True == False
base_models = [[ElasticNetCV(cv=10,n_jobs=-1), RidgeCV(cv=10, scoring='r2'), LassoLarsCV(cv=10,n_jobs=-1), ARDRegression(), BayesianRidge()][b]]
for base_model in base_models:
    for cat in numcat:
        for et in tqdm(numet):
            for lgbm in tqdm(numlgbm):
                for svr in numsvr:
                    for xgb in numxgb:
                        do = False
                        models_list = []
                        cv_targets_list = []
                        if svr:
                            do =True
                            for i in svr:
                                models_list.append(svr_list[i])
                                cv_targets_list.append(svr_targets_list[i])
                        if xgb:
                            for i in xgb:
                                models_list.append(xgb_list[i])
                                cv_targets_list.append(xgb_targets_list[i])
                                do =True
                                
                        if et:
                            do =True
                            for i in et:
                                models_list.append(et_list[i])
                                cv_targets_list.append(et_targets_list[i])
                        if cat:
                            do =True
                            for i in cat:
                                 
                                models_list.append(cat_list[i])
                                cv_targets_list.append(cat_targets_list[i])
                                
                                
                        if lgbm:
                            do =True
                            for i in lgbm:
                                models_list.append(lgbm_list[i])
                                cv_targets_list.append(lgbm_targets_list[i])
                        
                  
                        if do:

                            scores = []

                            for k in range(len(y_tests)):
                                # base_model = RidgeCV(alphas=[0.1, 1.0, 10.0], scoring='r2')

                                
                                k_models_list = []
                                k_cv_targets_list = []
                                k_meta_x_list = []
                                for i in range(len(models_list)):
                                    k_models_list.append(models_list[i][k])
                                    k_cv_targets_list.append(cv_targets_list[i][k][:, np.newaxis])

                                k_cv_targets = np.concatenate(k_cv_targets_list, axis=1)
                                
                                base_model.fit(k_cv_targets, y_train[k])
                                for model in k_models_list:
                                    k_meta_x_list.append(model.predict(X_test[k])[:,np.newaxis])
                                k_meta_x = np.concatenate(k_meta_x_list, axis=1)
                                scores.append(r2_score(y_tests[k], base_model.predict(k_meta_x)))

                            if max_score < np.mean(scores):
                                top_base = str(base_model)
                                max_score = np.mean(scores)
                                top_model = [cat, et, lgbm, svr, xgb, svr]
top = []
top.append(top_base)
top.append(str(max_score))
top.append(str(top_model))

textfile = open('top_combinations/'+args.data+'/'+str(args.num)+".txt", "w")
for element in top:
    textfile.write(element + "\n")
textfile.close()