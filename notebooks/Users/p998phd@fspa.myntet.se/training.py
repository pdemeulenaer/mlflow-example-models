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
import time as tm
import math
import json
import pandas as pd
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

choice_AIlab_or_local = 4

if choice_AIlab_or_local == 1:
    # If AIlab application (if not, comment out)
    import pydoop.hdfs as pydoop
    from hops import hdfs

if choice_AIlab_or_local !=4: from utils import *

# Spark/HIVE Variables
#global spark
spark = SparkSession\
    .builder\
    .appName("Cashflow 2")\
    .enableHiveSupport()\
    .getOrCreate()  # \
# .sparkContext.setLogLevel("ERROR")
# set the log level to one of ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE,
# WARN (default INFO)
spark.sparkContext.setLogLevel("ERROR")
#spark.conf.set("spark.sql.shuffle.partitions", "10")
# spark.conf.set("spark.dynamicAllocation.enabled","false")
# spark.conf.set("spark.shuffle.service.enabled","false")

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
    except:
        pass
    cwd = "/dbfs/mnt/test/"
else:
    cwd = os.getcwd()+"/"


# ===========================
# Reading configuration files
# ===========================

# Reading configuration files
data_conf = Get_Data_From_JSON(cwd + "data.json")
model_conf = Get_Data_From_JSON(cwd + "config.json")

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


# ---------------------------------------------------------------------------------------
# Main TRAINING Entry Point
# ---------------------------------------------------------------------------------------
def train(data_conf, model_conf, **kwargs):

    try:
        print("-----------------------------------")
        print("Starting Cashflow DL Model Training")
        print("-----------------------------------")
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
        # ========================================
        # T.1 Pre-processing before model training
        # ========================================

        # Loading dataset
        table_in = data_conf['synthetic_data']['table_to_train_on']
        if choice_AIlab_or_local == 1:
            ts_balance = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance = spark.read.parquet("{0}.parquet".format(table_in)).cache()
        if choice_AIlab_or_local == 3:
            ts_balance = spark.table("ddp_cvm.{0}".format(table_in)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()
            #ts_balance = spark.read.parquet(cwd_blob+"{0}.parquet".format(table_in)).cache()


        # ========================================================================
        # Creating the dataset on which we train (and test and validate) the model
        # ========================================================================

        ts_balance_model = ts_balance.sample(False, 0.7, seed=0) #now 0.7, but in real case would be 0.1 at best... or 0.05

        print('ts_balance_model.count()',ts_balance_model.count())

        # ====================================
        # Pre-processing before model training
        # ====================================

        ts_balance_model = pre_processing(ts_balance_model,
                                          end_date,
                                          spark,
                                          serving=False)
        ts_balance_model.show(3)

        print('ts_balance_model.rdd.getNumPartitions()',ts_balance_model.rdd.getNumPartitions())

        # Reducing number of partitions, that might have exploded after previous operations
        if choice_AIlab_or_local == 1:
            ts_balance_model = ts_balance_model.repartition(200)
        #if choice_AIlab_or_local == 2:
        #    ts_balance_model = ts_balance_model.repartition(200)
        ts_balance_model.show(3)

        # Saving prepared dataset
        table_out = 'cashflow_training_step1'
        if choice_AIlab_or_local == 1:
            ts_balance_model.write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 2:
            ts_balance_model.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_out))
        if choice_AIlab_or_local == 3:
            ts_balance_model.write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_out), mode='overwrite')
        if choice_AIlab_or_local == 4:
            ts_balance_model.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))

    except Exception as e:
        print("Errored on step T.1: pre-processing before model training")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ========================================
        # T.2 Generating TRAIN, VAL, TEST datasets
        # ========================================

        # ===========================
        # re-Loading of whole dataset
        # ===========================

        # Loading datasets
        table_model = 'cashflow_training_step1'
        if choice_AIlab_or_local == 1:
            ts_balance_model = spark.read.parquet("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_model)).cache()
        if choice_AIlab_or_local == 2:
            ts_balance_model = spark.read.parquet("{0}.parquet".format(table_model)).cache()
        if choice_AIlab_or_local == 3:
            ts_balance_model = spark.table("ddp_cvm.{0}".format(table_model)).cache()
        if choice_AIlab_or_local == 4:
            ts_balance_model = spark.read.parquet("/mnt/test/{0}.parquet".format(table_model)).cache()
        ts_balance_model.show(3)

        print('ts_balance_model.count()', ts_balance_model.count())
        print('ts_balance_model.rdd.getNumPartitions()', ts_balance_model.rdd.getNumPartitions())

        train_set, val_set, test_set = ts_balance_model.randomSplit([0.6, 0.2, 0.2], seed=12345)
        train_set.show(3)
        print('train_set.rdd.getNumPartitions(), val_set.rdd.getNumPartitions(), test_set.rdd.getNumPartitions()',
              train_set.rdd.getNumPartitions(), val_set.rdd.getNumPartitions(), test_set.rdd.getNumPartitions())

        if choice_AIlab_or_local == 1:
            train_set = train_set.repartition(800)
            val_set = val_set.repartition(800)
            test_set = train_set.repartition(800)
        #if choice_AIlab_or_local == 2:
        #    train_set = train_set.repartition(2000)
        #    val_set = val_set.repartition(2000)
        #    test_set = train_set.repartition(2000)
        #
        #print('train_set.rdd.getNumPartitions(), val_set.rdd.getNumPartitions(), test_set.rdd.getNumPartitions()',
        #      train_set.rdd.getNumPartitions(), val_set.rdd.getNumPartitions(), test_set.rdd.getNumPartitions())

        # Saving prepared datasets (train, val, test sets to parquet)
        table_train = 'cashflow_train'
        table_val = 'cashflow_val'
        table_test = data_conf['synthetic_data']['table_test_for_performance'] #'cashflow_test'
        if choice_AIlab_or_local == 1: # If AIlab
            train_set.select('X','y').write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_train))
            val_set.select('X','y').write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_val))
            #test_set.write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_test))
            test_set.select('primaryaccountholder','transactiondate','balance')\
                .write.format("parquet").mode("overwrite").save("hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_test))
        if choice_AIlab_or_local == 2: # If local
            train_set.select('X','y').write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_train))
            val_set.select('X','y').write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_val))
            #test_set.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_test))
            test_set.select('primaryaccountholder','transactiondate','balance')\
                .write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_test))
        if choice_AIlab_or_local == 3:
            train_set.select('X','y').write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_train), mode='overwrite')
            val_set.select('X','y').write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_val), mode='overwrite')
            #test_set.write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_test), mode='overwrite')
            test_set.select('primaryaccountholder','transactiondate','balance')\
                .write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_test), mode='overwrite')
        if choice_AIlab_or_local == 4:
            train_set.select('X','y').write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_train))
            val_set.select('X','y').write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_val))
            #test_set.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_test))
            test_set.select('primaryaccountholder','transactiondate','balance')\
                .write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_test))

    except Exception as e:
        print("Errored on step T.2: pre-processings")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ==============================
        # T.3 MODEL DEFINITION AND TRAIN
        # ==============================

        # ===========================
        # Model definition
        # ===========================
        tf.keras.backend.clear_session()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        model = define_1dcnn_model(N_days_X, N_days_y, model_conf)
        print(model.summary())

        # ========================================
        # Converting model to tensorflow estimator
        # ========================================

        if choice_AIlab_or_local == 1: # If AIlab application
            data_dir = hdfs.project_path()
            model_dir = pydoop.path.abspath(data_dir + "Models/{0}/".format(model_name))

        if choice_AIlab_or_local == 2: # If local application
            model_dir = os.path.join(cwd, model_name)#.replace("//", "\\")

        if choice_AIlab_or_local == 4: # If cloud application
            # Model checkpoints will be saved to the driver machine's local filesystem.
            model_dir = "/tmp/"+model_name
            print("model_dir: ", model_dir)
            # erase previous model, if exists, in the local folder
            dbutils.fs.rm("file:/tmp/{0}/".format(model_name), recurse=True)
            # erase previous model from the DBFS folder
            dbutils.fs.rm("dbfs:/mnt/test/{0}/".format(model_name), recurse=True)

        # run_config: https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
        run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            #log_device_placement=True,
            #save_checkpoints_steps=1000,
            save_summary_steps=10,
            #log_step_count_steps=50,
            #save_checkpoints_secs = 10 #20*60,  # Save checkpoints every 20 minutes. #incompatible with save_checkpoints_steps
            #keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
        )

        estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                          model_dir=model_dir,
                                                          config=run_config)

        # Loading datasets
        table_train = 'cashflow_train'
        table_val = 'cashflow_val'
        table_test = data_conf['synthetic_data']['table_test_for_performance']
        if choice_AIlab_or_local == 1:
            train_url = 'hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet'.format(table_train)
            val_url = 'hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet'.format(table_val)
            test_url = 'hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet'.format(table_test)
            input_files_train = pydoop.path.abspath(train_url)
            input_files_val = pydoop.path.abspath(val_url)
            input_files_test = pydoop.path.abspath(test_url)

        if choice_AIlab_or_local == 2:
            input_files_train = path_local+'{0}.parquet'.format(table_train)
            input_files_val = path_local+'{0}.parquet'.format(table_val)
            input_files_test = path_local+'{0}.parquet'.format(table_test)

        if choice_AIlab_or_local == 4:

            def get_local_path(dbfs_path):
                return os.path.join("/dbfs", dbfs_path.lstrip("/"))

            def whitelist_underscore_parquet_files(parquet_path):
                # In case of Databricks there is a bug reading parquet files:
                # Until ARROW-4723 is resolved we need to whitelist _* files that Databricks Runtime creates when saving data as Parquet.
                import pyarrow.parquet as pq
                underscore_files = [f for f in os.listdir(get_local_path(parquet_path)) if f.startswith("_")]
                pq.EXCLUDED_PARQUET_PATHS.update(underscore_files)
                return "file://" + get_local_path(parquet_path)

            input_files_train = whitelist_underscore_parquet_files('/mnt/test/{0}.parquet'.format(table_train))
            input_files_val = whitelist_underscore_parquet_files('/mnt/test/{0}.parquet'.format(table_val))
            input_files_test = whitelist_underscore_parquet_files('/mnt/test/{0}.parquet'.format(table_test))

        def transform_reader(reader, batch_size):
            def transform_input(x):
                inputs = tf.reshape(x.X, [-1,N_days_X,1])
                outputs = tf.reshape(x.y, [-1,N_days_y])
                return (inputs, outputs)
            return make_petastorm_dataset(reader).map(transform_input)\
                .apply(unbatch()).shuffle(3*batch_size, seed=42)\
                .batch(batch_size, drop_remainder=True)

           # Note that Petastorm produces Datasets that deliver data in batches that depends
           # entirely on the Parquet files' row group size. To control the batch size for
           # training, it's necessary to use Tensorflow's unbatch() and batch() operations
           # to re-batch the data into the right size. Also, note the small workaround that's
           # currently necessary to avoid a problem in reading Parquet files via Arrow in
           # Petastorm.

        #import math
        batch_size = int(model_conf['hyperParameters']['batch_size']) #200
        #steps_per_epoch = math.ceil(10000 / batch_size) #math.ceil(len(X_ready) / batch_size)
        num_epochs = int(model_conf['hyperParameters']['epochs']) #20
        #print(num_epochs * steps_per_epoch)

        #with make_batch_reader(input_files_train, hdfs_driver='libhdfs', num_epochs=num_epochs) as train_reader:
        #    with make_batch_reader(input_files_val, hdfs_driver='libhdfs', num_epochs=1) as val_reader:
        with make_batch_reader(input_files_train, num_epochs=num_epochs) as train_reader:
            with make_batch_reader(input_files_val, num_epochs=1) as val_reader:              

                def input_fn(reader,batch_size,time_steps_X,time_steps_y,feature_dim):
                    dataset = transform_reader(reader, batch_size)
                    dataset = dataset.prefetch(batch_size)
                    return dataset

                train_spec = tf.estimator.TrainSpec(
                    input_fn=lambda: input_fn(reader=train_reader,
                                              batch_size=batch_size,
                                              time_steps_X=N_days_X,
                                              time_steps_y=N_days_y,
                                              feature_dim=1,
                                             ),
                    max_steps=1e6)     #1e6) #num_epochs * steps_per_epoch == 200
                eval_spec = tf.estimator.EvalSpec(
                    input_fn=lambda: input_fn(reader=val_reader,
                                              batch_size=batch_size,
                                              time_steps_X=N_days_X,
                                              time_steps_y=N_days_y,
                                              feature_dim=1,
                                             ),
                    steps=None)

                start = tm.time()
                tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
                end = tm.time()
                print("Model training took %.2d s" % (end-start))
                latest = tf.train.latest_checkpoint(model_dir)
                print('latest',latest)

        # Saving the estimator to .pb file
        def serving_input_receiver_fn():
            # Inspired from https://guillaumegenthial.github.io/serving-tensorflow-estimator.html
            inputs = {
                model.input_names[0]: tf.placeholder(tf.float32, [None, N_days_X, 1]),
            }
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        export_dir = estimator.export_saved_model(model_dir+'/model', serving_input_receiver_fn)
        print('Estimator saved here: ', export_dir)

        if choice_AIlab_or_local == 4:
            # Copy the file from the driver node and save it to DBFS (so that they can be accessed e.g. after the current cluster terminates.):
            dbutils.fs.cp("file:/tmp/{0}/".format(model_name), "dbfs:/mnt/test/{0}/".format(model_name), recurse=True)
            # find file of model
            file = [f for f in os.listdir(get_local_path('/mnt/test/{0}/model/'.format(model_name)))]
            print('Estimator copied here: ', "dbfs:/mnt/test/{0}/model/".format(model_name)+file[0])

    except Exception as e:
        print("Errored on step T.3: model definition and train")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    train(data_conf, model_conf)

# COMMAND ----------

