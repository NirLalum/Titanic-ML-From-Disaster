# ===== Packages =====
import pandas as pd
from pandas.core.frame import DataFrame
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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