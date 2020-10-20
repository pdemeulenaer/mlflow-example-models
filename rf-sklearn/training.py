# Databricks notebook source
# Declaration of input parameter: the environment selected
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("environment", "TEST", 
                         ["TEST", "SYST", "PROD"], "The environment selected for run")

# COMMAND ----------

import os
import sys
import json
import socket
import traceback
import pandas as pd
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab

#Import of SKLEARN packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

import mlflow

# COMMAND ----------

data_json = '''{
    "TEST": {
        "input_train": "default.iris",
        "input_test": "default.iris_test",
        "output_test": "default.iris_test_scored",
        "input_to_score": "default.iris_to_score",
        "output_to_score": "default.scored"  
    },
    "SYST": {
        "input_train": "test.iris",
        "input_test": "test.iris_test",
        "output_test": "test.iris_test_scored",
        "input_to_score": "test.iris_to_score",
        "output_to_score": "test.scored"        
    },
    "PROD": {
        "input_train": "test.iris",
        "input_test": "test.iris_test",
        "output_test": "test.iris_test_scored",
        "input_to_score": "test.iris_to_score",
        "output_to_score": "test.scored"       
    }
}'''

config_json = '''{
    "hyperparameters": {
        "max_depth": "20",
        "n_estimators": "100",
        "max_features": "auto",
        "criterion": "gini",
        "class_weight": "balanced",
        "bootstrap": "True",
        "random_state": "21"        
    }
}'''

data_conf = json.loads(data_json)
model_conf = json.loads(config_json)

data_conf, model_conf

# COMMAND ----------

# Define the environment (dev, test or prod)
env = dbutils.widgets.getArgument("environment")

print()
print('Running in ', env)    

data_conf = json.loads(data_json)
model_conf = json.loads(config_json)

print(data_conf[env])
print(model_conf)

# Define the MLFlow experiment location
mlflow.set_experiment("/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment")


# ---------------------------------------------------------------------------------------
# Main TRAINING Entry Point
# ---------------------------------------------------------------------------------------
def train(data_conf, model_conf, **kwargs):

    try:
        print()
        print("-----------------------------------")
        print("         Model Training            ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

        # Loading of dataset
        iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
        idx = list(range(len(iris.target)))
        np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
        X = iris.data[idx]
        y = iris.target[idx]

        # Load data in Pandas dataFrame
        data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        data_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        data_pd.head()
        
        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target       = 'label'   
        
        X = data_pd[feature_cols].values
        y = data_pd[target].values

        # Creation of train and test datasets
        x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 
        
        # Save test dataset
        test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        test_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        test_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        test_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        test_df = spark.createDataFrame(test_pd)
        test_df.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format('test_data_sklearn_rf'))

        print("Step 1.0 completed: Loaded Iris dataset in Pandas")      

    except Exception as e:
        print("Errored on 1.0: data loading")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ========================================
        # 1.1 Model training
        # ========================================
        
        with mlflow.start_run() as run:          

            # Model definition
            max_depth = int(model_conf['hyperparameters']['max_depth'])
            n_estimators = int(model_conf['hyperparameters']['n_estimators'])
            max_features = model_conf['hyperparameters']['max_features']
            criterion = model_conf['hyperparameters']['criterion']
            class_weight = model_conf['hyperparameters']['class_weight']
            bootstrap = bool(model_conf['hyperparameters']['bootstrap'])
            clf = RandomForestClassifier(max_depth=max_depth,
                                       n_estimators=n_estimators,
                                       max_features=max_features,
                                       criterion=criterion,
                                       class_weight=class_weight,
                                       bootstrap=bootstrap,
                                       random_state=21,
                                       n_jobs=-1)          
            
            # Fit of the model on the training set
            model = clf.fit(x_train, y_train) 
            
            # Log the model within the MLflow run
            mlflow.log_param("max_depth", str(max_depth))
            mlflow.log_param("n_estimators", str(n_estimators))  
            mlflow.log_param("max_features", str(max_features))             
            mlflow.log_param("criterion", str(criterion))  
            mlflow.log_param("class_weight", str(class_weight))  
            mlflow.log_param("bootstrap", str(bootstrap))  
            mlflow.log_param("max_features", str(max_features)) 
            mlflow.sklearn.log_model(model, 
                                   "model",
                                   registered_model_name="sklearn-rf")                        

        print("Step 1.1 completed: model training and saved to MLFlow")                  

    except Exception as e:
        print("Errored on step 1.1: model training")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e       

    print()     

if __name__ == "__main__":
    train(data_conf, model_conf) 

# COMMAND ----------



# COMMAND ----------

import matplotlib.pyplot as plt  # doctest: +SKIP
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test,cmap='Blues')  # doctest: +SKIP
plt.show()  # doctest: +SKIP
plt.savefig('confusion_matrix.png')



# COMMAND ----------

#labels = ['business', 'health']
cm = confusion_matrix(y_test, y_test)#, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm,cmap='Blues')
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# COMMAND ----------

#FEATURES SELECTION
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target       = 'label'

def train(clf_name, clf, df_used, feature_cols, target):

      X = df_used[feature_cols].values
      y = df_used[target].values

      # CREATION OF TRAIN-TEST SETS
      x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets!

      # THE FIT ON THE TRAINING SET
      clf.fit(x_train, y_train)

      # THE CLASSIFICATION
      y_pred = clf.predict(x_test)

      # EVALUATION OF THE ACCURACY
      accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
      print('Results with the classifier: ', clf_name.upper())
      print('Accuracy derived (1=100%): ', accuracy)

      return y_test, y_pred

clf_name = 'Random Forest'
clf = RandomForestClassifier(#max_depth=50,
                           n_estimators=100,
                           max_features='auto',
                           criterion='gini',#'entropy',
                           class_weight='balanced',
                           bootstrap=True,
                           random_state=21,
                           n_jobs=-1) #n_jobs=-1 uses all available cores!!!

y_test, y_pred = train(clf_name, clf, data_pd, feature_cols, target)

#RANKING OF VARIABLES (available only for Random Forest)
print("Ranking of variables from Random Forest:")
feature_importance_index_sorted = np.argsort(clf.feature_importances_)[::-1]
for jj in feature_importance_index_sorted:
  print(feature_cols[jj],np.around(clf.feature_importances_[jj],decimals=3)*100,'%')

# Accuracy and Confusion Matrix
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy = ',accuracy)
#print 'ROC_AUC  = ', roc_auc
print('Confusion matrix:')
C = confusion_matrix(y_test, y_pred)
C_normalized = C / C.astype(np.float).sum()
#print C_normalized

Classes           = ['setosa','versicolor','virginica']
C_normalized_pd = pd.DataFrame(C_normalized,columns=Classes,index=Classes)
C_normalized_pd

# COMMAND ----------

        # Loading of dataset
        iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
        idx = list(range(len(iris.target)))
        np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
        X = iris.data[idx]
        y = iris.target[idx]

        # Load data in Pandas dataFrame
        data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        data_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        data_pd.head()
        
        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target       = 'label'   
        
        X = data_pd[feature_cols].values
        y = data_pd[target].values

        # Creation of train and test datasets
        x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets! 
        
        test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        test_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        test_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        test_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        test_df = spark.createDataFrame(test_pd)
        test_df.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format('test_data_sklearn_rf'))

# COMMAND ----------

        test_pd = pd.DataFrame(data=np.column_stack((x_test,y_test)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        test_pd.loc[data_pd['label']==0,'species'] = 'setosa'
        test_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
        test_pd.loc[data_pd['label']==2,'species'] = 'virginica'
        test_pd.head()

# COMMAND ----------

X = data_pd[feature_cols].values
X

# COMMAND ----------

