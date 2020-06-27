import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_single_dataset(ds_name, delimiter):
    df = pd.read_csv('UCI_datasets/' + ds_name, delimiter, header=None)
    elaborate_specific_dataset(df, ds_name)
    df = shuffle(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = pd.DataFrame(StandardScaler().fit_transform(X.values))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def elaborate_specific_dataset(df, ds_name):
    if ds_name == 'banknote.csv':
        df.iloc[:, len(df.columns) - 1].replace({0: -1}, inplace=True)
    elif ds_name == 'androgen.csv':
        cleanup_nums = {"positive": 1, "negative": -1}
        df.replace(cleanup_nums, inplace=True)
    elif ds_name == 'biodeg.csv':
        cleanup_nums = {"RB": 1, "NRB": -1}
        df.replace(cleanup_nums, inplace=True)
    else:
        print("Invalid dataset name")
