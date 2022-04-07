import pandas as pd
import numpy as np
from pickle5 import pickle
import seaborn as sns
from src import config
# DATA COLLECTION
from sklearn.datasets import load_boston
boston=load_boston()
df=pd.DataFrame(boston.data)
df.columns=boston.feature_names
df['Price']=boston.target
df.head()
df.to_csv(r'C:\Users\karti\PycharmProjects\linearRegressionBoston\data\data.csv')

# Feature Engineering
def correlated_features (data,threshold):
    correlated_columns=[]
    corr_matrix=data.corr()
    col=corr_matrix.columns
    ind=corr_matrix.index
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j] >= abs(threshold)) and i !=j :
                correlated_columns.append({'column':col[i], 'row': ind[j], 'value': corr_matrix.iloc[i,j]})
    print(correlated_columns)

correlated_features(df,0.8)

    # As RAD has corrlation > 0.9 with TAX variable, we are going to delete one of them..
def del_columns(data,column):
    data=data.drop(column, axis=1)
    data.to_csv(config.PROCESSED_DATA,index=False)
    print('column remove and file is saved')

del_columns(df,'TAX')


