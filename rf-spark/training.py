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

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as psf
import pyspark.sql.types as pst

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
        "fraction": "0.7",
        "numTrees": "12",
        "maxDepth": "10"
    }
}'''

data_conf = json.loads(data_json)
model_conf = json.loads(config_json)

data_conf, model_conf

# COMMAND ----------

# # -*- coding: utf-8 -*-

# # spark-submit --name simple_app --executor-memory 5G --master yarn --driver-memory 2G --executor-cores 5 --num-executors 3--conf spark.logConf=true simple_app.py 


# import os
# import sys
# import json
# import socket
# import traceback
# import pandas as pd
# import numpy as np

# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# from pyspark.sql.window import Window
# import pyspark.sql.functions as psf
# import pyspark.sql.types as pst

# from pyspark.ml import Pipeline
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from sklearn.datasets import load_iris

# spark = SparkSession\
#     .builder\
#     .appName("training")\
#     .enableHiveSupport()\
#     .getOrCreate()  

# # Detecting the environment
# def Set_Env():
#     '''
#     This function defines the environment on which the code is run, whether it is
#     TEST, SYST, PROD (based on the host name)
#     '''
#     global env
#     if any(s in str(socket.gethostname()) for s in ("y52951","y52953")):
#         env = "TEST"
#     if any(s in str(socket.gethostname()) for s in ("y78389","y75713")):        
#         env = "SYST"
#     if "sb-hdppra" in str(socket.gethostname()):
#         env = "PROD"

#     return env
        
# # Set Env
# Set_Env()   

# Define the environment (dev, test or prod)
env = dbutils.widgets.getArgument("environment")

print()
print('Running in ', env)    

# # Reading the configuration files
# def read_json(filename):
#     with open(filename, 'r') as stream:
#         json_dict = json.load(stream)
#         return json_dict

# model_conf = read_json( os.path.join('config', 'config.json') )
# data_conf = read_json( os.path.join('config', 'tables.json') )

data_conf = json.loads(data_json)
model_conf = json.loads(config_json)

print(data_conf[env])
print(model_conf)

# Define the MLFlow experiment location
mlflow.set_experiment("/Shared/simple-rf-spark/simple-rf-spark_experiment")

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

        # USING IRIS DATASET:
        iris = load_iris()                  #The Iris dataset is available through the scikit-learn API
        idx = list(range(len(iris.target)))
        np.random.shuffle(idx)              #We shuffle it (important if we want to split in train and test sets)
        X = iris.data[idx]
        y = iris.target[idx]

        # Load data in Pandas dataFrame and then in a Pyspark dataframe
        data_pd = pd.DataFrame(data=np.column_stack((X,y)), columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
        data_df = spark.createDataFrame(data_pd)

        #data_df = spark.table('test.iris')
        data_df.show(5)

        print("Step 1.0 completed: Loaded Iris dataset in Spark")      

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

            # This transforms the labels into indexes. See https://spark.apache.org/docs/latest/ml-features.html#stringindexer
            labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")

            # This groups all the features in one pack "features", needed for the VectorIndexer
            vectorizer = VectorAssembler(inputCols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],outputCol = "features")

            # This identifies categorical features, and indexes them. Set maxCategories so features with > 4 distinct values are treated as continuous. #https://spark.apache.org/docs/latest/ml-features.html#stringindexer
            featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)

            # Split the data into training and test sets (30% held out for testing)
            fraction = float(model_conf['hyperparameters']['fraction'])
            (trainingData, testData) = data_df.randomSplit([fraction, 1.-fraction])
            testData.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format('test_data_spark_rf'))

            # Train a RandomForest model. 
            numTrees = int(model_conf['hyperparameters']['numTrees'])
            maxDepth = int(model_conf['hyperparameters']['maxDepth'])
            rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=numTrees,  maxDepth=10)

            # Chain indexers and forest in a Pipeline
            pipeline = Pipeline(stages=[labelIndexer, vectorizer, featureIndexer, rf])

            # Train model.  This also runs the indexers.
            model = pipeline.fit(trainingData)

            # Save model to HDFS
            #model.save("rf_model")
            model.write().overwrite().save("mnt/rf_model_test")
            
            # Log the model within the MLflow run
            mlflow.log_param("fraction", str(fraction))
            mlflow.log_param("numTrees", str(numTrees))  
            mlflow.log_param("maxDepth", str(maxDepth))             
            mlflow.spark.log_model(model, 
                                   "mnt/spark-rf_model_test",
                                   registered_model_name="spark-rf")

        print("Step 1.1 completed: model training and saved to HDFS")                  

    except Exception as e:
        print("Errored on step 1.1: model training")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e       

    print()     

if __name__ == "__main__":
    train(data_conf, model_conf)    

# COMMAND ----------

#%fs ls dbfs:/tmp/mlflow

# COMMAND ----------

#%fs rm -r dbfs:/rf_model_test/

# COMMAND ----------

# from pyspark.ml import Pipeline
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.feature import HashingTF, Tokenizer
# training = spark.createDataFrame([
#     (0, "a b c d e spark", 1.0),
#     (1, "b d", 0.0),
#     (2, "spark f g h", 1.0),
#     (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
# tokenizer = Tokenizer(inputCol="text", outputCol="words")
# hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
# lr = LogisticRegression(maxIter=10, regParam=0.001)
# pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
# model = pipeline.fit(training)
# mlflow.spark.log_model(model, "spark-model")