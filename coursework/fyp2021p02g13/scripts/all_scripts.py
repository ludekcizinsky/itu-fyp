# Standard packages
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

# TASK 1
import seaborn as sns

# TASK 2
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# TASK 3
import folium
from IPython.core.display import display, HTML # Needed if you are using Deepnote



# =================== LOAD & SAVE DATA FUNCTIONS =======================


def load_data_numpy(filepath, encoding='utf', names=True, usemask=True, skip_header=0, usecols=None, max_rows=None, dtype=None, missing_values=None, delimiter=','):
    """
    Loads in CSV file using numpy
    :filename: full path to the given file, string
    :return: structured masked numpy array
    """

    data = np.genfromtxt(filepath,
                         delimiter=delimiter,
                         dtype=dtype,
                         names=names,
                         encoding=encoding,
                         usemask=usemask,
                         skip_header=skip_header,
                         usecols=usecols,
                         max_rows=max_rows,
                         missing_values=missing_values
                         )

    return data


def load_data_pandas(filepath, sep=','):

    """
    Loads data into a pandas dataframe.
    Note: It uses default expectations for missing values.
    See detailed description: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    :filepath: path to the file that you wan to load into pandas dataframe
    :sep: separator of values in the given file
    :return: pandas dataframe
    """

    df = pd.read_csv(filepath, sep=sep)
    return df


def load_json(filepath):
    """
    Loads a Json file into a python dictionary
    :filepath: path to the Json file
    :returns: python dictionary
    """
    
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def save_df_to_csv(df, path, index=False):

    """
    Saves pandas dataframe to csv file.
    See docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    :df: pandas dataframe
    :path: where to save the dataframe
    :return: None
    """

    df.to_csv(path, index=index)



# =================== MAP VISUALIZATION =======================
def folium_show(m, use_deepnote=False):
    """
    Short: Displays folium map.
    Long:
    This is a wrapper function which helps to handle the ENV problems - working within Deepnote VS working within standard jupyter
    notebook on a local machine. See: https://community.deepnote.com/c/forum/cannot-show-folium-map-on-deepnote
    :m: folium map object
    """
    if use_deepnote:
        data = m.get_root().render()
        data_fixed_height = data.replace('width: 100%;height: 100%', 'width: 100%').replace('height: 100.0%;', 'height: 609px;', 1)
        display(HTML(data_fixed_height))
    else:
        return m