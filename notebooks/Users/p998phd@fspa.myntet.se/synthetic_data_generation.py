# Databricks notebook source
# MAGIC %run /Users/p998phd@fspa.myntet.se/utils

# COMMAND ----------

spark

# COMMAND ----------

# ===============
# Packages import
# ===============

from __future__ import division
from datetime import datetime
import os
import random
import pandas as pd
import numpy as np
import logging
import yaml
from dateutil.relativedelta import relativedelta
from scipy import signal
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import udf

# ===============================
# Setting global variables
# ===============================

# 1: AIlab
# 2: Local application (laptop)
# 3: DDP/ODL
# 4: Azure cloud

choice_AIlab_or_local = 4
if choice_AIlab_or_local !=4: from utils import *   

# ===============================
# Definition of Spark application
# ===============================

if choice_AIlab_or_local == 2:
    import findspark
    findspark.init()
    findspark.find()
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[*]")\
                                .config("spark.driver.cores", 1)\
                                .config('spark.driver.memory','4g')\
                                .appName("myAppName")\
                                .getOrCreate()
    sc = spark.sparkContext
    
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "false")

if choice_AIlab_or_local == 1:    
    import pydoop.hdfs as pydoop
    from hops import hdfs
    
if choice_AIlab_or_local == 4:
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
    cwd = os.getcwd() + "/"

# COMMAND ----------

# ===========================
# Reading configuration files
# ===========================

# Reading configuration files
data_conf = Get_Data_From_JSON(cwd + "data.json")
model_conf = Get_Data_From_JSON(cwd + "config.json")

start_date, end_date = data_conf['start_date'], data_conf['end_date']
N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days'])  # 365, 92

end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
start_date_for_prediction_dt = end_date_dt - relativedelta(days=N_days_X + N_days_y)
start_date_for_prediction = start_date_for_prediction_dt.strftime("%Y-%m-%d")

start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(
    start_date, end_date, N_days_X, N_days_y)

time_range = pd.date_range(start_date, end_date, freq='D')

N_customers = 1e5 #2.5e6
serving_mode = False  # True if creating data for serving

# COMMAND ----------

# =========
# Functions
# =========

def time_series_generator(size=635,
                          cycle_period=30.5,
                          signal_type='sine',
                          salary=1,
                          trend=0.1,
                          noise=0.1,
                          offset=False,
                          spike=0):
    '''
    This function generates mock time series with noise
    :param (int) size: length of the time series
    :param (float) cycle_period: period of the signal (usually 30.5, the month period, in days)
    :param (string) signal_type: Type of signal, "sine", "sawtooth", "triangle", "square", "random_choice"
    :param (float) salary: Base scaling variable for the trend, default=1
    :param (float) trend: Scaling variable for the trend
    :param (float) noise: Trend noise, default=0.1
    :param (boolean) offset: Use of random phase offset, makes seasonality
    :param (int) spike: Number of random amplitude spikes
    :return (numpy array): Timeseries with account balance for each day
    '''

    signal_types = ['sine', 'sawtooth', 'triangle', 'square']
    if signal_type == 'random_choice':
        signal_type = random.choice(signal_types)
    elif signal_type not in signal_types:
        raise ValueError('{} is not a valid signal type'.format(signal_type))

    # in size = 635, and cycle_period = 30.5, we have ~ 21 periods (20.8)
    count_periods = size / cycle_period

    # 1. The trend making
    t = np.linspace(-0.5 * cycle_period * count_periods, 0.5 * cycle_period * count_periods, size)
    t_trend = np.linspace(0, 1, size)
    trend = trend * salary * t_trend ** 2

    # 2. The seasonality making
    if offset:
        phase = np.random.uniform(-1, 1) * np.pi
    else:
        phase = 0

    if signal_type == 'sine':     ts = 0.25 * salary * np.sin(2 * np.pi * (1. / cycle_period) * t + phase)
    if signal_type == 'sawtooth': ts = -0.25 * salary * signal.sawtooth(2 * np.pi * (1. / cycle_period) * t + phase)
    if signal_type == 'triangle': ts = 0.5 * salary * np.abs(
        signal.sawtooth(2 * np.pi * (1. / cycle_period) * t + phase)) - 1
    if signal_type == 'square':   ts = 0.25 * salary * signal.square(2 * np.pi * (1. / cycle_period) * t + phase)

    # 3. The noise making
    noise = np.random.normal(0, noise * salary, size)

    ts = ts + trend + noise

    # 4. Adding spikes to the time series
    if spike > 0:
        last_spike_time = int(size * 0.70)  # Don't create spikes at the end where we want to predict
        for _ in range(spike):
            sign = random.choice([-1, 1])
            t_spike = np.random.randint(0, last_spike_time)  # time of the spike
            ts[t_spike:] = ts[t_spike:] + sign * np.random.normal(3 * salary, salary)

    return ts

# COMMAND ----------

# ============================
# Generation of synthetic data (for both formats)
# ============================

dff = spark.range(N_customers).toDF("primaryaccountholder")

@udf("array<float>") 
def ts_generation():
    bb = time_series_generator(
              size=len(time_range),
              cycle_period=30.5,
              signal_type='random_choice',
              salary=np.maximum(np.random.normal(15000, 5000), 100),
              trend=np.random.normal(0, 3),
              noise=np.abs(np.random.normal(0, 0.01)) + 0.05,
              offset=True,
              spike=1).tolist()      
    return np.around(bb,decimals=2).tolist()
    
dff = dff.withColumn("balance", ts_generation())

dff2 = spark.sql("SELECT sequence(to_date('{0}'), to_date('{1}'), interval 1 day) as transactiondate".format(start_date, end_date))

timeseries_spark = dff2.crossJoin(dff)
timeseries_spark = timeseries_spark.select('primaryaccountholder','transactiondate','balance')

timeseries_spark.show(5)
timeseries_spark.count()

# COMMAND ----------

# ========================
# Saving the dataset
# ========================

if not serving_mode:
    table_out = data_conf['synthetic_data']['table_to_train_on'] #'cashflow_ts_mock_10K_ts'
else:    
    table_out = data_conf['synthetic_data']['table_to_score'] #'cashflow_ts_mock_10K_ts_heldout'

if choice_AIlab_or_local == 1:
    timeseries_spark.write.format("parquet").mode("overwrite").save(
        "hdfs:///Projects/Cashflow_AMMF/Cashflow_AMMF_Training_Datasets/{0}.parquet".format(table_out))
if choice_AIlab_or_local == 2:
    timeseries_spark.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_out))
if choice_AIlab_or_local == 3:
    timeseries_spark.write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_out), mode='overwrite')
if choice_AIlab_or_local == 4:
    try:
        dbutils.fs.mount(
            source="wasbs://blob1@storageaccountcloudai.blob.core.windows.net",
            mount_point="/mnt/test",
            extra_configs={"fs.azure.account.key.storageaccountcloudai.blob.core.windows.net": dbutils.secrets.get(
                scope="key-vault-secrets-cloudai", key="storageaccountcloudaiKey1")})
    except:
        pass
    timeseries_spark.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out)) 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# UNNECESSARY STUFF

# COMMAND ----------

ddddd = spark.sql("SELECT sequence(to_date('{0}'), to_date('{1}'), interval 1 day) as transactiondate".format(start_date, end_date))

ddd = spark.range(N_customers).toDF("primaryaccountholder")

timeseries_spark = ddddd.crossJoin(ddd)
#timeseries_spark = timeseries_spark.select('primaryaccountholder','transactiondate','balance')

timeseries_spark.show(5)

# COMMAND ----------

    end_date_plusOneDay = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=1)
    end_date_plusOneDay = end_date_plusOneDay.strftime("%Y-%m-%d")
    
    end_date_plus92Day = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=92)
    end_date_plus92Day = end_date_plus92Day.strftime("%Y-%m-%d")
    end_date_plus92Day
    
    end_date_plus92Day = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=92)
    end_date_plus92Day = end_date_plus92Day.strftime("%Y-%m-%d") 
    extrapolated_spark_df = spark.sql("SELECT sequence(to_date('{0}'), to_date('{1}'), interval 1 day) as transactiondate".format(end_date_plusOneDay, end_date_plus92Day))    
    
    extrapolated_spark_df.show()

# COMMAND ----------

extrapolated_spark_df.dtypes

# COMMAND ----------

time_range = pd.date_range(start=end_date_plusOneDay, periods=92)
#time_range

# COMMAND ----------

    ddd = spark.range(N_customers).toDF("primaryaccountholder")
    
    # extrapolation time
    time_range = pd.date_range(start=end_date_plusOneDay, periods=92)
    extrapolated_df = pd.DataFrame(columns=['transactiondate_next3months'])
    extrapolated_df['transactiondate_next3months'] = time_range
    extrapolated_df['transactiondate_next3months'] = extrapolated_df['transactiondate_next3months'].astype(str)

    extrapolated_spark_df = spark.sparkContext.parallelize([
        [extrapolated_df['transactiondate_next3months'].tolist()],
    ]).toDF(['transactiondate_next3months'])
    extrapolated_spark_df.show()
    
    #bidon = ddd.crossJoin(extrapolated_spark_df)
    #bidon.show()
    
    extrapolated_spark_df.dtypes
    

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Deprecated after this 

# COMMAND ----------

# =======================
# Generation of fake data (for both formats)
# =======================

time_range = pd.date_range(start_date, end_date, freq='D')
balance_df = pd.DataFrame(np.zeros((len(time_range), N_customers)),
                          columns=list(range(N_customers)))  # Dim = number_days x number_of_customers

trend_array = np.random.normal(0, 3, N_customers)
salary_array = np.maximum(
    np.random.normal(
        15000, 5000, N_customers), np.full(
        (N_customers), 100))
noise_array = np.abs(np.random.normal(0, 0.01, N_customers)) + 0.05

for column in balance_df.columns:
    balance_df[column] = time_series_generator(
        size=len(time_range),
        cycle_period=30.5,
        signal_type='random_choice',
        salary=salary_array[column],
        trend=trend_array[column],
        noise=noise_array[column],
        offset=True,
        spike=1)

timeseries_all = timeseries_all.round(2)    
timeseries_all = balance_df.copy()
timeseries_all.set_index(time_range, inplace=True)
timeseries_all.reset_index(inplace=True)
timeseries_all.rename(columns={'index': 'date'}, inplace=True)
timeseries_all['date'] = pd.to_datetime(timeseries_all['date'])

# Converting column names to integers
columns_original = timeseries_all.columns  # if you want to keep original column names
timeseries_all.columns = [timeseries_all.columns[0]] + list(
    range(len(timeseries_all.columns[1:])))  # TODO: Check if this is necessary

timeseries_all = timeseries_all.round(2) 

# COMMAND ----------

timeseries_all.head()

# COMMAND ----------

# =================
# Format conversion (to be fed to spark format: primaryaccountholder transactiondate balance)
# =================

balance_df_for_spark = timeseries_all.drop('date', axis=1)
balance_df_for_spark.set_index(time_range, inplace=True)
balance_df_for_spark = balance_df_for_spark.unstack().reset_index().loc[:, ['level_0', 'level_1', 0]]
balance_df_for_spark.columns = ['primaryaccountholder', 'transactiondate', 'balance']
balance_df_for_spark['transactiondate'] = pd.to_datetime(balance_df_for_spark['transactiondate'])

# COMMAND ----------

balance_df_for_spark.head()

# COMMAND ----------

# CONVERSION TO SPARK USING ARROW
timeseries_spark = spark.createDataFrame(balance_df_for_spark)
timeseries_spark = timeseries_spark.withColumn('transactiondate', F.to_date(F.col('transactiondate')))
timeseries_spark = timeseries_spark.withColumn('balance', F.round(F.col('balance'), 3))

# COMMAND ----------

timeseries_spark.show()

# COMMAND ----------










# ================================
# Converting to time series format
# ================================

w = Window.partitionBy('primaryaccountholder').orderBy(
    F.asc('transactiondate'))

timeseries_spark = timeseries_spark.withColumn(
    'transactiondate_list',
    F.collect_list('transactiondate').over(w))
timeseries_spark = timeseries_spark.withColumn(
    'balance_list',
    F.collect_list('balance').over(w))

# We get many duplicate rows for each customer, picking only one
timeseries_spark = timeseries_spark.groupBy('primaryaccountholder').agg(
    F.max('transactiondate_list').alias('transactiondate'),
    F.max('balance_list').alias('balance'))

# ========================
# Saving the dataset
# ========================

if not serving_mode:
    table_out = 'cashflow_ts_mock_10K_ts'
else:    
    table_out = 'cashflow_ts_mock_10K_ts_heldout'

if choice_AIlab_or_local == 1:
    timeseries_spark.write.format("parquet").mode("overwrite").save(
        "hdfs:///Projects/Cashflow_prediction/Cashflow_prediction_Training_Datasets/{0}.parquet".format(table_out))
if choice_AIlab_or_local == 2:
    timeseries_spark.write.format("parquet").mode("overwrite").save("{0}.parquet".format(table_out))
if choice_AIlab_or_local == 3:
    timeseries_spark.write.format("parquet").saveAsTable('ddp_cvm.{0}'.format(table_out), mode='overwrite')
if choice_AIlab_or_local == 4:
    try:
        dbutils.fs.mount(
            source="wasbs://blob1@storageaccountcloudai.blob.core.windows.net",
            mount_point="/mnt/test",
            extra_configs={"fs.azure.account.key.storageaccountcloudai.blob.core.windows.net": dbutils.secrets.get(
                scope="key-vault-secrets-cloudai", key="storageaccountcloudaiKey1")})
    except:
        pass
    timeseries_spark.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# This is a completely different way, really more optimized, as in spark from start

# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType, DoubleType, DateType, StringType
from pyspark.sql.functions import udf

dff = spark.range(2.5e6).toDF("id")

@udf("array<float>") 
def ts_generation():
    bb = time_series_generator(
              size=len(time_range),
              cycle_period=30.5,
              signal_type='random_choice',
              salary=np.maximum(np.random.normal(15000, 5000), 100),
              trend=np.random.normal(0, 3),
              noise=np.abs(np.random.normal(0, 0.01)) + 0.05,
              offset=True,
              spike=1).tolist()      
    return np.around(bb,decimals=2).tolist()
    
dff = dff.withColumn("balance", ts_generation())

dddddd = spark.sql("SELECT sequence(to_date('{0}'), to_date('{1}'), interval 1 day) as transactiondate".format(start_date, end_date))
#dddddd.show()

final = dddddd.crossJoin(dff)
final = final.select('id','transactiondate','balance')

final.show()

# COMMAND ----------

final.count()

# COMMAND ----------

