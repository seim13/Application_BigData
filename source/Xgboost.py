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


# ## Xgboost

# In[21]:


import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import seaborn as sns


# In[25]:


#allow logloss and classification error plots for each iteraetion of xgb model
"""def plot_compare(metrics,eval_results,epochs):
    for m in metrics:
        test_score = eval_results['val'][m]
        train_score = eval_results['train'][m]
        rang = range(0, epochs)
        plt.rcParams["figure.figsize"] = [6,6]
        plt.plot(rang, test_score,"c", label="Val")
        plt.plot(rang, train_score,"orange", label="Train")
        title_name = m + " plot"
        plt.title(title_name)
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.show()"""
        
"""def fitXgb(sk_model, training_data=training_data,epochs=300):
    print('Fitting model...')
    sk_model.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    print('Fitting done!')
    train = xgb.DMatrix(training_data['X_train'], label=training_data['Y_train'])
    val = xgb.DMatrix(training_data['X_val'], label=training_data['Y_val'])
    params = sk_model.get_xgb_params()
    metrics = ['mlogloss','merror']
    params['eval_metric'] = metrics
    store = {}
    evallist = [(val, 'val'),(train,'train')]
    xgb_model = xgb.train(params, train, epochs, evallist,evals_result=store,verbose_eval=100)
    print('-- Model Report --')
    print('XGBoost Accuracy: '+str(accuracy_score(sk_model.predict(training_data['X_test']), training_data['Y_test'])))
    print('XGBoost F1-Score (Micro): '+str(f1_score(sk_model.predict(training_data['X_test']),training_data['Y_test'],average='micro')))
    plot_compare(metrics,store,epochs)
    
    plot_compare(metrics,store,epochs)
    features = column_nan
    f, ax = plt.subplots(figsize=(10,5))
    plot = sns.barplot(x=features, y=sk_model.feature_importances_)
    ax.set_title('Feature Importance')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.show()"""


# In[26]:


#initial model
with mlflow.start_run():
    xgb1 = XGBClassifier(learning_rate=0.1,
                        n_estimators=500,
                        max_depth=10,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='multi:softmax',
                        nthread=4,
                        num_class=9,
                        seed=27)


    xgb1.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

    #train = xgb1.DMatrix(training_data['X_train'], label=training_data['Y_train'])
    #val = xgb1.DMatrix(training_data['X_val'], label=training_data['Y_val'])
    #params = sk_model.get_xg1_params()
    #metrics = ['mlogloss','merror']
    #params['eval_metric'] = metrics
    


    mlflow.log_param("random_state",  27)
    
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth",10)
    mlflow.log_param("min_child_weight",1)
    mlflow.log_param("subsample",0.8)
    mlflow.log_param("nthread",4)
    mlflow.log_param("num_class", 9)
    mlflow.log_param("seed", 27)

   
    mlflow.log_metric("accuracy", accuracy_score(xgb1.predict(training_data['X_test']), training_data['Y_test']))
    mlflow.log_metric("f1_score", f1_score(xgb1.predict(training_data['X_test']),training_data['Y_test'],average='micro'))

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(xgb1, "model", registered_model_name="XGboost")
    else:
        mlflow.sklearn.log_model(xgb1, "XGboost")
    
    


# ## Gradient boosting

