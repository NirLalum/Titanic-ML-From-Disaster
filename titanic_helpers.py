# ===== Packages =====
import pandas as pd
from pandas.core.frame import DataFrame
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prettytable import PrettyTable


plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = True


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
    table = PrettyTable(field_names = ['Feature', 'Missing values in train', 'Missing values in test'])
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
