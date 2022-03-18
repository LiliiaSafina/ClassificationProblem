import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoding_for_column(df, col_name):
    enc = OneHotEncoder()
    enc.fit(df[col_name].values.reshape(-1, 1))
    df[enc.get_feature_names_out([col_name])] = enc.transform(df[col_name].values.reshape(-1, 1)).toarray()


def df_only_float(df):
    columns_for_oht = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
    for col in columns_for_oht:
        one_hot_encoding_for_column(df, col)
        df.drop(columns=col, inplace=True)


def result_column(df):
    res_class = np.zeros(shape=[df.shape[0], 1])
    for i in range(df.shape[0]):
        if df['class'][i] == ' <=50K.' or df['class'][i] == ' <=50K':
            res_class[i] = 1
        else:
            res_class[i] = 2
    df['class'] = res_class


def native_country(df):
    res_class = np.zeros(shape=[df.shape[0], 1])
    for i in range(df.shape[0]):
        if df['native-country'][i] == 'United-States':
            res_class[i] = 2
        else:
            res_class[i] = 1
    df['native-country'] = res_class


def get_data_frame(filename):
    df = pd.read_csv(filename, delimiter=',', names=[
                                                        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                                        'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                                        'capital-gain', 'capital-loss', 'hours-per-week',
                                                        'native-country', 'class'
                                                        ]
                     )
    df_only_float(df)
    native_country(df)
    result_column(df)
    return df


def get_data(filename):
    df = pd.read_csv(filename, header=0)
    x = df.drop(columns=['class'])
    y = df['class']
    return x, y


def write_to_file(df, filename):
    df.to_csv(filename, columns=df.columns, index=False)

