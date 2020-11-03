#!/usr/bin/env python
# coding: utf-8

# # Model building For balanced dataset

# ## Prepared dataset

# In[11]:


import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import pickle



df_balanced=pd.read_csv('df_balanced.csv')
df_test=pd.read_csv('df_clean_test.csv')


# In[3]:


#df_test.drop('Unnamed: 0', axis=1, inplace=True)


# In[31]:


df_balanced


# In[37]:


column_nan=df_balanced.columns.tolist()


# In[38]:


column_nan


# In[33]:


column_test=df_test.columns.tolist()




def stratified_split(df, target, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs


train_idxs, val_idxs = stratified_split(df_balanced, 'TARGET', val_percent=0.25)

val_idxs, test_idxs = stratified_split(df_balanced[df_balanced.index.isin(val_idxs)], 'TARGET', val_percent=0.5)


# In[40]:


train_df = df_balanced[df_balanced.index.isin(train_idxs)]

X_train = train_df[column_nan].values
Y_train = train_df[['TARGET']].values
print('Retrieved Training Data')
print(X_train.shape,'----',Y_train.shape)

val_df = df_balanced[df_balanced.index.isin(val_idxs)]
X_val = val_df[column_nan].values
Y_val = val_df[['TARGET']].values
print('Retrieved Validation Data')
print(X_val.shape,'----',Y_val.shape)

test_df = df_balanced[df_balanced.index.isin(test_idxs)]
X_test = test_df[column_nan].values
Y_test = test_df[['TARGET']].values
print('Retrieved Test Data')
print(X_test.shape,'----',Y_test.shape)


# In[41]:


#store data, all in numpy arrays
training_data = {'X_train':X_train,'Y_train':Y_train,
                'X_val': X_val,'Y_val':Y_val,
                'X_test': X_test,'Y_test':Y_test}


# ## Random Forest





import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import seaborn as sns




from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix





# In[32]:


with mlflow.start_run():

    gb_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_features=2, max_depth=12, random_state=0)
    gb_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0]))



# In[33]:

    predicted_labels = gb_clf.predict(training_data['X_test'])


# In[35]:



    mlflow.log_param("random_state",  0)
    mlflow.log_param("max_depth", 12)
    mlflow.log_param("n_estimators", 1000)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy_train", gb_clf.score(X_train, Y_train))
    mlflow.log_metric("accuracy_validation", gb_clf.score(X_val, Y_val))
    mlflow.log_metric("accuracy_score", accuracy_score(training_data['Y_test'], predicted_labels))


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(gb_clf, "model", registered_model_name="GradientBoosting")
    else:
        mlflow.sklearn.log_model(gb_clf, "GradientBoosting")





