import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from scipy import stats
from autofeat import FeatureSelector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def cap_outliers(train, test, method, cap):
    for col in train.columns:
        if method == "median":
            cent_stat = train[col].median()
            dev_stat = np.abs(stats.median_abs_deviation(train[col]))
        else:  # method == "mean" or method == "nan"
            cent_stat = train[col].mean()
            dev_stat = train[col].std()

        lower_threshold = cent_stat - cap * dev_stat
        upper_threshold = cent_stat + cap * dev_stat
        if method != "nan":
            train[col] = train[col].apply(lambda x: min(max(lower_threshold, x), upper_threshold))
            test[col] = test[col].apply(lambda x: min(max(lower_threshold, x), upper_threshold))
        else:  # method == "nan"
            train[col] = train[col].apply(lambda x: x if lower_threshold < x < upper_threshold else np.nan)
            test[col] = test[col].apply(lambda x: x if lower_threshold < x < upper_threshold else np.nan)
    return train, test


def import_and_preprocess(imp_method=None,
                          outlier_method=None,
                          outlier_cap=None,
                          standardize_method=None,
                          feature_selection_method=None,
                          n_features=None):
    base = '/cluster/work/riner/patakiz/AML/'
    x_test = pd.read_csv(base + "X_test.csv")
    x_train = pd.read_csv(base + "X_train.csv")
    y_train = pd.read_csv(base + "y_train.csv")

    # Drop indexes
    x_test.drop(columns=["id"], inplace=True)
    y_train.drop(columns=["id"], inplace=True)
    x_train.drop(columns=["id"], inplace=True)

    # Drop homogenous columns
    homogenous_columns = []
    for column_name in x_train.columns:
        if x_train[column_name].nunique() == 1:
            homogenous_columns.append(column_name)
    x_train.drop(homogenous_columns, inplace=True, axis=1)
    x_test.drop(homogenous_columns, inplace=True, axis=1)

    # Handle outliers
    if outlier_method in ["mean", "median", 'nan']:
        x_train, x_test = cap_outliers(x_train, x_test, method=outlier_method, cap=outlier_cap)

    train_mean, train_median = x_train.mean(), x_train.median()

    # NaN imputation
    if imp_method in ["mean", "median"]:
        x_train_imp = train_median if imp_method == "median" else train_mean
        x_train.fillna(x_train_imp, inplace=True)
        x_test.fillna(x_train_imp, inplace=True)
    elif imp_method == "knn":
        imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
        x_train = imputer.fit_transform(x_train, y_train)
        x_test = imputer.transform(x_test)
    elif imp_method == "iterative":
        imputer = IterativeImputer(random_state=0)
        x_train = imputer.fit_transform(x_train, y_train)
        x_test = imputer.transform(x_test)

    # Normalize all data
    if standardize_method == "standardize":
        train_std = x_train.std()
        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std
    elif standardize_method == "normalize":
        train_min, train_max = x_train.min(), x_train.max()
        x_train = (x_train - train_min) / (train_max - train_min)
        x_test = (x_test - train_min) / (train_max - train_min)

    # Feature selection
    if feature_selection_method == "pca":
        pca = PCA(n_components=n_features)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
    elif feature_selection_method == "tsne":
        tsne = TSNE(n_components=n_features, verbose=1, method='exact')
        x_train = tsne.fit_transform(x_train)
        x_test = tsne.transform(x_test)
    elif feature_selection_method == "autofeat":
        fsel = FeatureSelector(verbose=1)
        x_train = fsel.fit_transform(x_train, y_train)
        x_test = fsel.transform(x_test)
    elif feature_selection_method == "kbest":
        test = SelectKBest(score_func=f_classif, k=n_features)
        fit = test.fit(x_train, y_train)
        x_train = fit.transform(x_train)
        x_test = fit.transform(x_test)

    return x_train, y_train, x_test
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", help='method', type=str, required=True)
args = parser.parse_args()
#["mean", "median", "nan"]
if 'mean' in args.method:
    outlier_methods = ['mean']
elif 'median' in args.method:
    outlier_methods = ['median']
elif 'nan' in args.method:
    outlier_methods = ['nan']

if '1' in args.method:outlier_caps = list(range(3, 7, 1))
elif '2' in args.method:outlier_caps = list(range(7, 11, 1))

# outlier_methods = ["mean"]
# outlier_caps = [5]


search = [(method, cap) for method in outlier_methods for cap in outlier_caps]
results = []

for method, cap in search:
    print(f"Running for {method}, {cap}")
    x_train, y_train, x_test = import_and_preprocess(imp_method="mean",
                                                     outlier_method=method,
                                                     outlier_cap=cap,
                                                     feature_selection_method="autofeat")
    print("Data processed")
    model = CatBoostRegressor(n_estimators=9080, verbose=0)
    score = np.mean(cross_val_score(model, x_train, y_train, cv=10, n_jobs=-1, scoring='r2'))
    results.append(f"Method: {method}\tCap: {cap}\tScore:{score}")

with open(args.method+".txt", "w+") as file:
    for item in results:
        file.write(item + '\n')
