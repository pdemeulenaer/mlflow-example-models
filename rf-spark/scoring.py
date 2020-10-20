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

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as psf
import pyspark.sql.types as pst

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

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
        "fraction": "0.7",
        "numTrees": "12",
        "maxDepth": "10"
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

mlflow_model_name = 'spark-rf' #model_conf['model_name']

# Define the scoring mode
# evaluate_or_score = str(sys.argv[1])
evaluate_or_score = dbutils.widgets.getArgument("evaluate_or_score")

# Define the MLFlow experiment location
mlflow.set_experiment("/Shared/simple-rf-spark/simple-rf-spark_experiment")


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

#         # USING IRIS DATASET:
#         iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
#         idx = list(range(len(iris.target)))
#         np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
#         X = iris.data[idx]
#         y = iris.target[idx]

#         # Load data in Pandas dataFrame and then in a Pyspark dataframe
#         data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
#         data_df = spark.createDataFrame(data_pd)

        #if not evaluation: table_in = data_conf[env]['input_to_score'] # for scoring new data
        #if evaluation: table_in = data_conf[env]['input_test'] # for performance evaluation on historical data
        #data_df = spark.table(table_in)
        data_df = spark.read.format("delta").load("/mnt/delta/{0}".format('test_data_spark_rf'))        
 
        data_df.show(5)
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
        #model = PipelineModel.load("/tmp/rf_model_test")        
        model = mlflow.spark.load_model(mlflow_path)        
        
        # Make predictions
        predictions = model.transform(data_df)

        # Select example rows to display.
        predictions.select("prediction", "indexedLabel", "features").show(5)

        # Saving the result of the scoring
        if not evaluation: table_out = data_conf[env]['output_to_score']
        if evaluation: table_out = data_conf[env]['output_test']
        #predictions.write.format("ORC").saveAsTable(table_out, mode='overwrite')        

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
        table_in = data_conf[env]['output_test']
        predictions = spark.table(table_in) 
        
        # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Accuracy = %g" % (accuracy))
        
        # Extracting the test set to Pandas
        pred_pd = predictions.toPandas()          
        y_test = pred_pd['indexedLabel'].values
        y_pred = pred_pd['prediction'].values     
        
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
        fig.savefig('/dbfs/mnt/delta/confusion_matrix_spark_rf.png')

        # Tracking performance metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_artifact("/dbfs/mnt/delta/confusion_matrix_spark_rf.png")         

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

