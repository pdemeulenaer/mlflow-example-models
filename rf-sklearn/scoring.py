# Databricks notebook source
# Declaration of input parameter: the environment selected
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("environment", "TEST", 
                         ["TEST", "SYST", "PROD"], "The environment selected for run")

dbutils.widgets.dropdown("evaluate_or_score", "score", 
                         ["score", "evaluate"], "The mode of the scoring: simple scoring of unseen data (score) or score on test dataset (evaluate)")

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
from sklearn.metrics import plot_confusion_matrix

import mlflow
from mlflow.tracking.client import MlflowClient

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

mlflow_model_name = 'sklearn-rf' #model_conf['model_name']

# Define the scoring mode
# evaluate_or_score = str(sys.argv[1])
evaluate_or_score = dbutils.widgets.getArgument("evaluate_or_score")

# Define the MLFlow experiment location
mlflow.set_experiment("/Shared/simple-rf-sklearn/simple-rf-sklearn_experiment")


# ---------------------------------------------------------------------------------------
# Main SCORING Entry Point
# ---------------------------------------------------------------------------------------
def score(data_conf, model_conf, evaluation=False, **kwargs):

    try:
        print()
        print("-----------------------------------")
        print("         Model Serving             ")
        print("-----------------------------------")
        print()

        # ==============================
        # 1.0 Data Loading
        # ==============================

#         # Loading of dataset
#         iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
#         idx = list(range(len(iris.target)))
#         np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
#         X = iris.data[idx]
#         y = iris.target[idx]

#         # Load data in Pandas dataFrame
#         data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
#         data_pd.loc[data_pd['label']==0,'species'] = 'setosa'
#         data_pd.loc[data_pd['label']==1,'species'] = 'versicolor'
#         data_pd.loc[data_pd['label']==2,'species'] = 'virginica'
#         data_pd.head()
        
        #if not evaluation: table_in = data_conf[env]['input_to_score'] # for scoring new data
        #if evaluation: table_in = data_conf[env]['input_test'] # for performance evaluation on historical data
        #data_df = spark.table(table_in)
        data_df = spark.read.format("delta").load("/mnt/delta/{0}".format('test_data_sklearn_rf'))  
        data_pd = data_df.toPandas()
        
        # Feature selection
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target       = 'label'   
        
        x_test = data_pd[feature_cols].values
        y_test = data_pd[target].values

        # Creation of train and test datasets
        #x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.7, stratify=y) #stratify=y ensures that the same proportion of labels are in both train and test sets!        
         
        print("Step 1.0 completed: Loaded dataset in Spark")      

    except Exception as e:
        print("Errored on 1.0: data loading")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===================
        # 1.1 Model serving
        # ===================   
        
        # Load model from MLflow model registry #https://www.mlflow.org/docs/latest/model-registry.html        
        if env == 'PROD' : 
            mlflow_model_stage = 'Production'
        else:
            mlflow_model_stage = 'Staging'
            
        print(mlflow_model_stage)
            
        # Detecting the model dictionary among available models in MLflow model registry. 
        client = MlflowClient()
        for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
            if dict(mv)['current_stage'] == mlflow_model_stage:
                model_dict = dict(mv)
                break  
                
        print('Model extracted run_id: ', model_dict['run_id'])
        print('Model extracted version number: ', model_dict['version'])
        print('Model extracted stage: ', model_dict['current_stage'])                

        def get_local_path_from_dbfs(dbfs_path):
            '''
            This get the local version of the dbfs path, i.e. replaces "dbfs:" by "/dbfs", for local APIs use.
            ''' 
            return "/dbfs"+dbfs_path.lstrip("dbfs:")  
      
        mlflow_path = model_dict['source']      
        print("mlflow_path: ", mlflow_path)        

        # De-serialize the model
        model = mlflow.sklearn.load_model(mlflow_path)        
        
        # Make predictions
        y_pred = model.predict(x_test)

        # Saving the result of the scoring
        if not evaluation: table_out = data_conf[env]['output_to_score']
        if evaluation: table_out = data_conf[env]['output_test']
        #predictions.write.format("ORC").saveAsTable(table_out, mode='overwrite') 
        pred_pd = pd.DataFrame(data=np.column_stack((y_test,y_pred)), columns=['y_test', 'y_pred'])
        pred_df = spark.createDataFrame(pred_pd)
        pred_df.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format('prediction_sklearn_rf'))  
        
        # Select example rows to display.
        pred_df.show(5)        

        print("Step 1.1 completed: model loading, data scoring and writing to hive")   
        print()               

    except Exception as e:
        print("Errored on step 1.1: model serving")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e   


def evaluate(data_conf, model_conf, scoring=True, **kwargs):

    try:
        # ===========================
        # E.1 Scoring of test data
        # ===========================
        if scoring: # switch, in case we want to skip score (if score already computed earlier)
            score(data_conf, model_conf, evaluation=True) # the score function is applied on test dataset for performance evaluation

    except Exception as e:
        print("Errored on step E.1: scoring of test data")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===========================
        # E.2 Metrics & Visualization
        # ===========================
        
        # Load model from MLflow model registry #https://www.mlflow.org/docs/latest/model-registry.html        
        if env == 'PROD' : 
            mlflow_model_stage = 'Production'
        else:
            mlflow_model_stage = 'Staging'
            
        # Detecting the model dictionary among available models in MLflow model registry. 
        client = MlflowClient()
        for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
            if dict(mv)['current_stage'] == mlflow_model_stage:
                model_dict = dict(mv)
                break     
                
        print('Model extracted run_id: ', model_dict['run_id'])
        print('Model extracted version number: ', model_dict['version'])
        print('Model extracted stage: ', model_dict['current_stage'])                            
         
        #MLflow logging of metrics for trained model
        mlflow.end_run() # in case mlfow run_id defined before here
        mlflow.start_run(run_id=model_dict['run_id'])        

        # Loading dataset
        #table_in = data_conf[env]['output_test']
        #predictions = spark.table(table_in)    
        pred_df = spark.read.format("delta").load("/mnt/delta/{0}".format('prediction_sklearn_rf'))  
        pred_pd = pred_df.toPandas() 
        y_test = pred_pd['y_test'].values
        y_pred = pred_pd['y_pred'].values      
        
        # Accuracy and Confusion Matrix
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy = ',accuracy)
        print('Confusion matrix:')
        Classes = ['setosa','versicolor','virginica']
        C = confusion_matrix(y_test, y_pred)
        C_normalized = C / C.astype(np.float).sum()        
        C_normalized_pd = pd.DataFrame(C_normalized,columns=Classes,index=Classes)
        print(C_normalized_pd)
        
        #labels = ['business', 'health']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(C,cmap='Blues')
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + Classes)
        ax.set_yticklabels([''] + Classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        fig.savefig('/dbfs/mnt/delta/confusion_matrix_sklearn_rf.png')

        # Tracking performance metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_artifact("/dbfs/mnt/delta/confusion_matrix_sklearn_rf.png")       

        print("Step E.2 completed metrics & visualisation")
        print()

    except Exception as e:
        print("Errored on step E.2: metrics & visualisation")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":

    print(evaluate_or_score)

    # If we want to score NEW (unseen) data:
    if evaluate_or_score == 'score':
        score(data_conf, model_conf, evaluation=False)
    if evaluate_or_score == 'evaluate':
        evaluate(data_conf, model_conf, scoring=True)  

# COMMAND ----------

