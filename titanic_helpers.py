# ===== PACKAGES =====
import pandas as pd
from pandas.core.frame import DataFrame
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from prettytable import PrettyTable

from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = True


# ----- FUNCTIONS -----
def value_counts_all(df: DataFrame):
    """Use pandas value_counts function for each column in the Dataframe
    Parameters
    ----------
    data: A pandas DataFrame
    Returns
    -------
    None
    """

    for col in df.columns:
        print('-' * 40 + col + '-' * 40 , end=' - ')
        print('\n')
        print(f"{df[col].value_counts()}\n")


def missing_values_table(df_train, df_test):
    table = PrettyTable(field_names = ['Feature', 'Missing values in train', 'Missing values in test'], )
    for feature in df_test.columns:
        table.add_row([feature, len(df_train[df_train[feature].isnull()]), len(df_test[df_test[feature].isnull()])])
    
    table.add_row(['Survived', len(df_train[df_train['Survived'].isnull()]), 'feature is NA'])
    print(table)


def detect_outliers(df, feature):
    Q1, Q3 = np.percentile(df[feature].values, 25), np.percentile(df[feature].values, 75)
    LQR = Q3-Q1
    step = 1.5*LQR
    lower_whisker = df[df[feature]>(Q1-step)][feature].min()
    upper_whisker = df[df[feature]<(Q3+step)][feature].max()

    return df.loc[(df[feature]<lower_whisker) | (df[feature]>upper_whisker)]


def encoding(df, labels_to_encode):
    # Encode
    cat_var = df[labels_to_encode]
    cat_var_dummies = pd.get_dummies(cat_var, drop_first=True)
    df.drop(labels_to_encode, axis=1, inplace=True)
    df = pd.concat([df, cat_var_dummies], axis=1)
    
    return df


def data_split(dataset:DataFrame ,n_splits=1, test_size=0.2, train_size=0.8):
    """ split the data into train and test (or train and validation)
    Parameters
    ----------
    dataset: A pandas Dataframe to split
    n_splits: (optional) An integer of number of splits
    test_size: (optional) An inegeger for the propotion of the test set
    train_size: (optional) An inegeger for the propotion of the train set
    Returns
    -------
    
    X_train, y_train, X_test, y_test
    """
    
    spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size)
    X = dataset.drop('Survived', axis=1)
    y = dataset['Survived']
    train_index, test_index = next(spliter.split(X, y))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, y_train, X_test, y_test


def model_evaluation(y_true, y_pred):
    """Evaluate the ML model according to different metrics
    Parameters
    ----------
    y_true: Data structure containing the true labels of the examples
    y_pred: Data structure containing the prediction of the ML model
    Returns
    -------
    None
    """

    metric_table = PrettyTable(float_format='.3', field_names = ['Metric', 'Score'])
    metric_table.add_row(['accuracy', accuracy_score(y_true, y_pred)])
    metric_table.add_row(['recall', recall_score(y_true, y_pred)])
    metric_table.add_row(['presicion', precision_score(y_true, y_pred)])
    
    print(metric_table)

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    disp.ax_.set_xticklabels(['negative','positive'])
    disp.ax_.set_yticklabels(['negative','positive'])