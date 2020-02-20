# Databricks notebook source
# MAGIC %run /Users/p998phd@fspa.myntet.se/utils

# COMMAND ----------

# -*- coding: utf-8 -*-

# MODEL GCVM0002

import os
import sys
import traceback
import logging
import logging.config
import yaml
import json
import time
import pandas as pd
from pandas import Series
import numpy as np
import random
import warnings
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.contrib import predictor
from tensorflow.contrib.data import unbatch
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf,PandasUDFType
import pyspark.sql.functions as F

choice_AIlab_or_local = 4

if choice_AIlab_or_local == 1:
    # If AIlab application (if not, comment out)
    import pydoop.hdfs as pydoop
    from hops import hdfs

if choice_AIlab_or_local !=4: from utils import *

# Spark/HIVE Variables
global spark
spark = SparkSession\
    .builder\
    .appName("Cashflow 2")\
    .enableHiveSupport()\
    .getOrCreate()  # \
# .sparkContext.setLogLevel("ERROR")
# set the log level to one of ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE,
# WARN (default INFO)
spark.sparkContext.setLogLevel("ERROR")
# spark.conf.set("spark.dynamicAllocation.enabled","false")
# spark.conf.set("spark.shuffle.service.enabled","false")

if choice_AIlab_or_local == 4:
    # Enable Arrow-based columnar data transfers
    spark.conf.set("spark.sql.shuffle.partitions", "1000")
    spark.conf.set("spark.default.parallelism", "1000") # this is valid only for RDDs
    spark.conf.set("spark.databricks.io.cache.enabled", "false")

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "false")

spark.conf.set("spark.task.maxFailures", "1000")
spark.conf.set("spark.blacklist.enabled", "true")

spark.conf.set("spark.yarn.maxAppAttempts", "1000")

spark.conf.set("spark.sql.shuffle.partitions", "1000")
spark.conf.set("spark.default.parallelism", "1000") # this is valid only for RDDs

if choice_AIlab_or_local == 4:

    # For reading from blob directly
    blob_name = "blob2"
    account_name = "storageaccountcloudai"
    storageKey1 = dbutils.secrets.get(scope = "key-vault-secrets-cloudai", key = "storageaccountcloudaiKey1")
    spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storageKey1)
    cwd_blob = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/"

    # For reading from mount point (if not already mounted)
    try:
        dbutils.fs.mount(
            source = "wasbs://blob1@storageaccountcloudai.blob.core.windows.net",
            mount_point = "/mnt/test",
            extra_configs = {"fs.azure.account.key.storageaccountcloudai.blob.core.windows.net":dbutils.secrets.get(scope = "key-vault-secrets-cloudai", key = "storageaccountcloudaiKey1")})
        print('Mount succeeded')
    except:
        print('Mount point already there')
    cwd = "/dbfs/mnt/test/"
else:
    cwd = os.getcwd()+"/"

# Reading configuration files
data_conf = Get_Data_From_JSON(cwd + "data.json")
model_conf = Get_Data_From_JSON(cwd + "config.json")



# ---------------------------------------------------------------------------------------
# Main SERVING Entry Point
# ---------------------------------------------------------------------------------------

def score(data_conf, model_conf, evaluation=False, **kwargs):

    try:
        print("----------------------------------")
        print("Starting Cashflow DL Model Scoring")
        print("----------------------------------")
        print()

        # ==============================
        # 0. Main parameters definitions
        # ==============================

        # Size of X and y arrays definition
        N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days']) #365, 92
        print('Number of days used for prediction (X): {0}'.format(N_days_X))
        print('Number of days predicted (y): {0}'.format(N_days_y))
        print('')

        # Date range definition
        start_date, end_date = data_conf['start_date'], data_conf['end_date']
        start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(start_date, end_date, N_days_X, N_days_y)
        print('Date range: [{0}, {1}]'.format(start_date, end_date))
        print('')

        if choice_AIlab_or_local == 2:
            path_local = 'file://'+cwd

        model_name = model_conf['model_name']

        #print("Step 0 completed (main parameters definition)")

    except Exception as e:
        print("Errored on initialization")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e
    '''
    try:
        # ==================================
        # S.1 Pre-processings before serving
        # ==================================

        start_time_S1 = time.time()

        # Loading dataset
        if not evaluation: table_in = data_conf['synthetic_data']['table_to_score'] # for scoring new data
        if evaluation: table_in = data_conf['synthetic_data']['table_test_for_performance'] # for performance evaluation on historical data
        if choice_AIlab_or_local == 1:
            ts_balance = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance = spark.read.parquet("{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 3:
            ts_balance = spark.table("ddp_cvm.{0}".format(table_in)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()
            #ts_balance = spark.read.parquet(cwd_blob+"{0}.parquet".format(table_in)).cache()

        #print('Reading table {0}'.format(table_in))
        #print('Size of table: ',ts_balance.count())
        #print('ts_balance.rdd.getNumPartitions()',ts_balance.rdd.getNumPartitions())

        #if choice_AIlab_or_local == 1:
        #    ts_balance = ts_balance.repartition(1000)

        if not evaluation:
            ts_balance = pre_processing(ts_balance, end_date, spark, serving=True)
        if evaluation:
            ts_balance = pre_processing(ts_balance, end_date, spark, serving=False)
        ts_balance.show(3)

        #print('Size of table: ',ts_balance.count())
        #print('Before serialization')
        #print('ts_balance.rdd.getNumPartitions()',ts_balance.rdd.getNumPartitions())

        # Reducing number of partitions, that might have exploded after previous operations
        #if choice_AIlab_or_local == 1:
        #    ts_balance = ts_balance.repartition(1000)
        #if choice_AIlab_or_local == 2:
        #    ts_balance = ts_balance.repartition(200)
        #if choice_AIlab_or_local == 3:
        #    ts_balance = ts_balance.repartition(2100)
        #if choice_AIlab_or_local == 4:
        #    ts_balance = ts_balance.repartition(1000)
        #ts_balance.show(3)

        # Saving prepared dataset
        if not evaluation: table_out = data_conf['synthetic_data']['cashflow_s1_out_scoring'] # for scoring new data
        if evaluation: table_out = 'cashflow_s1_out_evaluation' # for performance evaluation on historical data
        if choice_AIlab_or_local == 1:
            ts_balance.write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 2:
            ts_balance.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 3:
            ts_balance.write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_out), mode='overwrite')
        if choice_AIlab_or_local == 4:
            ts_balance.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))

        ts_balance.unpersist()
        spark.catalog.clearCache()
        end_time_S1 = time.time()
        print("Step S.1 completed: pre-processings before serving")
        print("Time spent: ", end_time_S1-start_time_S1)

    except Exception as e:
        print("Errored on step S.1: pre-processings before serving")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e
    '''
    try:
        # ===================
        # S.2 Model serving
        # ===================

        start_time_S2 = time.time()

        # Loading dataset
        if not evaluation: table_in = data_conf['synthetic_data']['cashflow_s1_out_scoring'] # for scoring new data
        if evaluation: table_in = 'cashflow_s1_out_evaluation' # for performance evaluation on historical data
        if choice_AIlab_or_local == 1:
            ts_balance = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance = spark.read.parquet("{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 3:
            ts_balance = spark.table("ddp_cvm.{0}".format(table_in)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in))
            #ts_balance = spark.read.format("delta").load("/mnt/test/{0}_delta".format(table_in))

        #ts_balance = ts_balance.select('primaryaccountholder','X')
        ts_balance.cache()
        #ts_balance.show(3)
        #print('Size of table: ',ts_balance.count())
        print('Number of  partitions: ', ts_balance.rdd.getNumPartitions())
        #ts_balance = ts_balance.repartition(1000)

        #dbutils.fs.cp("dbfs:/mnt/test/{0}/".format(model_name), "file:/tmp/{0}/".format(model_name), recurse=True)
        #from shutil import copyfile
        #copyfile("dbfs:/mnt/test/{0}/".format(model_name), '/dbfs/mnt/test/{0}/model/'.format(model_name))

        #Important! When serving on RDDs, the function is parallelized, so any other arguments than the array to serve should be
        #either defined statically or defined inside the function. Otherwise, it won't be parallelized... [Davit]
        #Inspired from https://www.oipapio.com/question-3285678

        #Interesting: finding latest model: https://nheise.com/tensorflow/2019/08/18/import-tf-estimator-without-tf-contrib.html

        def rdd_scoring(numpy_array):
            predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = export_dir_saved)
            return predictor_fn({'input': numpy_array.reshape(-1, N_days_X, 1) })

        if choice_AIlab_or_local == 1:
            #'hdfs://10.8.31.100:8020/Projects/Cashflow_prediction/Models/tf_models4//model/1578661004'
            #file = [f for f in os.listdir('/dbfs/mnt/test/{0}/model/'.format(model_name))]
            def rdd_scoring(numpy_array):
                if not os.path.exists(os.getcwd() + "/tf_models"):
                    hdfs.copy_to_local(hdfs.project_path() + "Models/tf_models",overwrite=True)
                export_dir_saved = 'tf_models4/model/1578661004'
                predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = export_dir_saved)
                return predictor_fn({'input': numpy_array.reshape(-1, N_days_X, 1) })

            @F.udf("array<float>")
            def udf_scoring(x):
                return np.around(rdd_scoring(np.array(x))['output'][0].tolist(), decimals=3).tolist()

            @F.pandas_udf("array<float>")
            def pandas_udf_scoring(x):
                #if not os.path.exists(hdfs.project_path() + "Models/tf_models4"):
                if not os.path.exists(os.getcwd() + "/tf_models4"):
                    hdfs.copy_to_local(hdfs.project_path() + "Models/tf_models4",overwrite=True)
                export_dir_saved = 'tf_models4/model/1578661004'
                predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = export_dir_saved)
                return x.apply(lambda v: np.around(predictor_fn({'input': np.array(v).reshape(-1, N_days_X, 1)})['output'][0].tolist(), decimals=3).tolist() )

        if choice_AIlab_or_local == 2:
            export_dir_saved = path_local+'tf_models4/model/1578573399'
                  
        if choice_AIlab_or_local == 4:
            # It detects the name id of the pb model file  
            file = [f for f in os.listdir('/dbfs/mnt/test/{0}/model/'.format(model_name))]
            export_dir_saved = "/dbfs/mnt/test/{0}/model/".format(model_name)+file[0]
            
            @F.pandas_udf("array<float>")
            def pandas_udf_scoring(x):
                predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = export_dir_saved)
                return x.apply(lambda v: np.around(predictor_fn({'input': np.array(v).reshape(-1, N_days_X, 1)})['output'][0].tolist(), decimals=3).tolist() )            

        ts_balance = ts_balance.withColumn('y_pred', pandas_udf_scoring('X'))
        ts_balance = ts_balance.withColumn('y_pred', ts_balance.y_pred.cast("array<float>"))
        print('ts_balance.rdd.getNumPartitions()',ts_balance.rdd.getNumPartitions())
        ts_balance.show(3)
        #print('Size of table: ',ts_balance.count())

        # Saving prepared dataset
        if not evaluation: table_out = data_conf['synthetic_data']['cashflow_s2_out_scoring'] # for scoring new data
        if evaluation: table_out = 'cashflow_s2_out_evaluation' # for performance evaluation on historical data
        if choice_AIlab_or_local == 1:
            ts_balance.write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 2:
            ts_balance.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 3:
            ts_balance.write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_out), mode='overwrite')
        if choice_AIlab_or_local == 4:
            ts_balance.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))
            #ts_balance.write.format("delta").mode("overwrite").save("/mnt/test/{0}_delta".format(table_out))

        ts_balance.unpersist()
        spark.catalog.clearCache()
        end_time_S2 = time.time()
        print("Step S.2 completed: model serving")
        print("Time spent: ", end_time_S2-start_time_S2)

    except Exception as e:
        print("Errored on step S.2: model serving")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===================
        # S.3 Post-processing
        # ===================

        start_time_S3 = time.time()

        # Loading dataset
        if not evaluation: table_in = data_conf['synthetic_data']['cashflow_s2_out_scoring'] # for scoring new data
        if evaluation: table_in  = 'cashflow_s2_out_evaluation' # for performance evaluation on historical data
        if choice_AIlab_or_local == 1:
            ts_balance = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance = spark.read.parquet("{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()

        ts_balance = post_processing(ts_balance)
        ts_balance.show(3)

        # Saving prepared dataset
        if not evaluation: table_out = data_conf['synthetic_data']['table_scored']
        if evaluation: table_out = data_conf['synthetic_data']['table_test_for_performance_scored']
        if choice_AIlab_or_local == 1:
            ts_balance.write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 2:
            ts_balance.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 4:
            ts_balance.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))

        ts_balance.unpersist()
        end_time_S3 = time.time()
        print("Step S.3 completed: post-processing")
        print("Time spent: ", end_time_S3-start_time_S3)

    except Exception as e:
        print("Errored on step S.3: post-processing")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e


def evaluate(data_conf, model_conf, scoring=True, **kwargs):

    #if kwargs['do_we_score'] is True: # switch, in case we want to skip score (if score already computed earlier)
    if scoring: # switch, in case we want to skip score (if score already computed earlier)
        score(data_conf, model_conf, evaluation=True) # the score function is applied on test dataset for performance evaluation

    try:
        print("-------------------------------------")
        print("Starting Cashflow DL Model Evaluation")
        print("-------------------------------------")
        print()

        # ==============================
        # 0. Main parameters definitions
        # ==============================

        # Size of X and y arrays definition
        N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days']) #365, 92
        print('Number of days used for prediction (X): ', N_days_X)
        print('Number of days predicted (y): ', N_days_y)
        print()

        # Date range definition
        start_date, end_date = data_conf['start_date'], data_conf['end_date']
        start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(start_date, end_date, N_days_X, N_days_y)
        print('Date range: ', start_date, end_date)
        print()

        if choice_AIlab_or_local == 2:
            path_local = 'file://'+cwd

        model_name = model_conf['model_name']

    except Exception as e:
        print("Errored on initialization")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===========================
        # S.4 Metrics & Visualization
        # ===========================

        # Loading dataset
        table_in = data_conf['synthetic_data']['table_test_for_performance_scored']
        if choice_AIlab_or_local == 1:
            ts_balance = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance = spark.read.parquet("{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()

        # Extracting the test set to Pandas
        ts_balance_pd = ts_balance.select('balance','X', 'y','y_pred','y_pred_rescaled_retrended').toPandas()

        # Extraction of metrics
        R2_all_3month, R2_array_3month, R2_all_1month, R2_array_1month = metric_extraction(ts_balance_pd, N_days_y)

        # Visualization of prediction
        visualization_prediction(ts_balance_pd, start_date, end_date, N_days_X, N_days_y, R2_array_1month, R2_array_3month, choice_AIlab_or_local, serving=False)

        # Saving the metric
        print('Test R2 metric (3-months window): {}'.format(R2_all_3month))
        print('Test R2 metric (1-months window): {}'.format(R2_all_1month))

        with open("/dbfs/mnt/test/evaluation.json", "w+") as f:
            json.dump({'R2_3month': R2_all_3month, 'R2_1month': R2_all_1month}, f)

        ts_balance.unpersist()
        print("Step S.4 completed visualisation")

    except Exception as e:
        print("Errored on step S.4: visualisation")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

# COMMAND ----------

if __name__ == "__main__":
    # If we want to score NEW (unseen) data:
    #score(data_conf, model_conf, evaluation=False)
    evaluate(data_conf, model_conf)

# COMMAND ----------

if __name__ == "__main__":
    # If we want to score NEW (unseen) data:
    #score(data_conf, model_conf, evaluation=False)
    evaluate(data_conf, model_conf)

# COMMAND ----------

    try:
        print("-------------------------------------")
        print("Starting Cashflow DL Model Evaluation")
        print("-------------------------------------")
        print()

        # ==============================
        # 0. Main parameters definitions
        # ==============================

        # Size of X and y arrays definition
        N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days']) #365, 92
        print('Number of days used for prediction (X): ', N_days_X)
        print('Number of days predicted (y): ', N_days_y)
        print()

        # Date range definition
        start_date, end_date = data_conf['start_date'], data_conf['end_date']
        start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(start_date, end_date, N_days_X, N_days_y)
        print('Date range: ', start_date, end_date)
        print()

        if choice_AIlab_or_local == 2:
            path_local = 'file://'+cwd

        model_name = model_conf['model_name']

    except Exception as e:
        print("Errored on initialization")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===========================
        # S.4 Metrics & Visualization
        # ===========================

        # Loading dataset
        table_in = data_conf['synthetic_data']['table_test_for_performance_scored']
        if choice_AIlab_or_local == 1:
            ts_balance = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance = spark.read.parquet("{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()

        # Extracting the test set to Pandas
        ts_balance_pd = ts_balance.select('balance','X', 'y','y_pred','y_pred_rescaled_retrended').toPandas()

        # Extraction of metrics
        R2_all_3month, R2_array_3month, R2_all_1month, R2_array_1month = metric_extraction(ts_balance_pd, N_days_y)

        # Visualization of prediction
        visualization_prediction(ts_balance_pd, start_date, end_date, N_days_X, N_days_y, R2_array_1month, R2_array_3month, choice_AIlab_or_local, serving=False)

        # Saving the metric
        print('Test R2 metric (3-months window): {}'.format(R2_all_3month))
        print('Test R2 metric (1-months window): {}'.format(R2_all_1month))

        with open("/dbfs/mnt/test/evaluation.json", "w+") as f:
            json.dump({'R2_3month': R2_all_3month, 'R2_1month': R2_all_1month}, f)

        ts_balance.unpersist()
        print("Step S.4 completed visualisation")

    except Exception as e:
        print("Errored on step S.4: visualisation")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

# COMMAND ----------

R2_array_3month.shape

# COMMAND ----------

#ts_balance_pd

# COMMAND ----------

visualization_prediction(ts_balance_pd, start_date, end_date, N_days_X, N_days_y, choice_AIlab_or_local, serving=False)