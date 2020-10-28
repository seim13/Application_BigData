#!/usr/bin/env python
# coding: utf-8

# # Application of BigData Part 1

# ## Model building For balanced dataset

# In[32]:


import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[79]:


df_balanced=pd.read_csv('df_balanced.csv')
df_test=pd.read_csv('df_clean_test.csv')




# In[81]:


column=df_balanced.columns.tolist()


# In[82]:


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


# In[83]:


train_df = df_balanced[df_balanced.index.isin(train_idxs)]

X_train = train_df[column].values
Y_train = train_df[['TARGET']].values
print('Retrieved Training Data')

val_df = df_balanced[df_balanced.index.isin(val_idxs)]
X_val = val_df[column].values
Y_val = val_df[['TARGET']].values
print('Retrieved Validation Data')

test_df = df_balanced[df_balanced.index.isin(test_idxs)]
X_test = test_df[column].values
Y_test = test_df[['TARGET']].values
print('Retrieved Test Data')


# In[84]:


#store data, all in numpy arrays
training_data = {'X_train':X_train,'Y_train':Y_train,
                'X_val': X_val,'Y_val':Y_val,
                'X_test': X_test,'Y_test':Y_test}


# In[26]:
with mlflow.start_run():

    clf_balanced = RandomForestClassifier(random_state=27,max_depth = 12, n_estimators = 200)
    clf_balanced.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))


    # In[28]:


    predicted_labels = clf_balanced.predict(training_data['X_test'])


    # In[29]:


    accuracy = accuracy_score(training_data['Y_test'], predicted_labels)


    mlflow.log_param("random_state",  27)
    mlflow.log_param("max_depth", 12)
    mlflow.log_param("n_estimators", 200)
   
    mlflow.log_metric("accuracy", accuracy)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(clf_balanced, "model", registered_model_name="RandomForestClassifier")
    else:
        mlflow.sklearn.log_model(clf_balanced, "RandomForestClassifier")

    # In[30]:

"""
    params = {
        'n_estimators'      : range(100,500,50),
        'max_depth'         : [8, 9, 10, 11, 12],
        'max_features': ['auto'],
        'criterion' :['gini']
    }
    #metrics to consider: f1_micro, f1_macro, roc_auc_ovr
    gsearch1 = GridSearchCV(estimator = clf, param_grid = params, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
    gsearch1.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))


# In[34]:


def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best


# In[35]:


getTrainScores(gsearch1)


# In[38]:


clf2 = gsearch1.best_estimator_

params1 = {
    'n_estimators'      : range(90,110,10),
    'max_depth'         : [9, 10,11]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch2 = GridSearchCV(estimator = clf2, param_grid = params1, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch2.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))


# In[39]:


getTrainScores(gsearch2)


# In[88]:


clf2.predict(X_test[4,:].reshape(1,-1))


# In[89]:





# In[ ]:





# In[40]:


import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import seaborn as sns


# In[47]:


#allow logloss and classification error plots for each iteraetion of xgb model
def plot_compare(metrics,eval_results,epochs):
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
        plt.show()
        
def fitXgb(sk_model, training_data=training_data,epochs=300):
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
    features = column
    f, ax = plt.subplots(figsize=(10,5))
    plot = sns.barplot(x=features, y=sk_model.feature_importances_)
    ax.set_title('Feature Importance')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.show()


# In[48]:


#initial model
xgb1 = XGBClassifier(learning_rate=0.1,
                    n_estimators=500,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=9,
                    seed=27)


# In[49]:


fitXgb(xgb1, training_data)

"""