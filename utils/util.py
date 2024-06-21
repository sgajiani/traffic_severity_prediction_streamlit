import os
import pandas as pd
import numpy as np


def author():
    return 'Samir Gajiani'


def get_file_path(filename):
    data_folder_path = os.environ.get('DATA_FOLDER_PATH')
    file_path = os.path.join(data_folder_path, filename)
    return file_path



def get_missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_pct = round(df.isnull().sum() / len(df) * 100, 2)
    df_mis_val = pd.concat((df.dtypes, mis_val, mis_val_pct), axis=1).reset_index()
    df_mis_val = df_mis_val.rename(columns={'index': 'col', 0: 'dtype', 1:'miss_val', 2:'pct_miss_val'})
    df_mis_val = df_mis_val[df_mis_val.iloc[:,2] != 0]
    df_mis_val = df_mis_val.sort_values('pct_miss_val', ascending=False).reset_index(drop=True)
    return df_mis_val



def get_unique_val(cat_col):
    return round(cat_col.value_counts(dropna=False) / len(cat_col) * 100, 2)


def get_max_class(df):
    columns = ['col', 'class', 'pct']
    df_max_class = pd.DataFrame(columns=columns)

    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            sr_pct = round(df[col].value_counts() / len(df[col]) * 100, 2)
            max_class, max_pct = sr_pct.idxmax(), sr_pct.max()
            max_tup = (col, max_class, max_pct)
            df_max_class = df_max_class.append(pd.Series(max_tup, index=columns), ignore_index=True)
    return df_max_class



def imbalance_check(df):
    df_miss_val = get_missing_values(df)
    df_max_class = get_max_class(df)
    df_imb = pd.merge(df_max_class, df_miss_val, on='col', how='left')
    df_imb['pct_miss_val'] = df_imb['pct_miss_val'].fillna(0.0)
    df_imb['impute_mode_pct'] = df_imb['pct'] + df_imb['pct_miss_val']
    df_imb = df_imb[['col', 'class', 'pct', 'pct_miss_val', 'impute_mode_pct']]
    df_imb = df_imb.sort_values('impute_mode_pct', ascending=False).reset_index(drop=True)
    return df_imb


def ordinal_encode_test(input_val, features): 
    feature_val = list(np.arange(len(features)))
    feature_key = features
    feature_dict = dict(zip(feature_key, feature_val))
    encoded_val = feature_dict[input_val]
    return encoded_val


def get_prediction(data, model):
    dict_response = {0: 'Fatal injury', 1: 'Serious Injury', 2: 'Slight Injury'}
    y_pred = model.predict(data)[0]
    return dict_response[y_pred]
