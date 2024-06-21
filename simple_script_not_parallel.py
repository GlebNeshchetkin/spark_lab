import time
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
import gc
import psutil

from catboost import CatBoostRegressor
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def preprocess_single_row(row, label_encoders):
    if row['year'].startswith("\ufeff"):
        row['year'] = row['year'][len("\ufeff"):]
    for col in categorical_cols:
        if col in row:
            value = row[col]
            if value in set(label_encoders[col].keys()):
                row[col] = label_encoders[col][value]
            else:
                row[col] = -1
    row = row.astype(float)
    return row

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem


model_path = "/output/catboost_model.bin"
catboost_regressor = CatBoostRegressor()
catboost_regressor.load_model(model_path)
spark = SparkSession.builder \
    .appName("My App 1") \
    .getOrCreate()
with open('/output/label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
time_vec = []
ram_vec = []
for iter in range(100):
    start = time.time()
    initial_memory_usage = get_memory_usage()
    df = spark.read.csv("hdfs://namenode:9001/car_prices_3.csv",header=True)
    pdf = df.toPandas()
    pdf.dropna(inplace=True)
    features = ['year','make','model','trim','body','transmission','state','condition','odometer','color','interior','seller','mmr']
    categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller']
    averaged_error = 0
    for i in pdf.index:
        data_loaded = pdf.loc[i, features]
        preprocessed_data = list(preprocess_single_row(data_loaded,label_encoders))
        predicted_price = catboost_regressor.predict(preprocessed_data)
        real_price = pdf.loc[i,'sellingprice']
        real_price = float(real_price)
        averaged_error += np.abs(real_price-float(predicted_price))
    averaged_error = averaged_error/len(pdf)
    end = time.time()
    time_vec.append(end-start)
    final_memory_usage = get_memory_usage()
    ram_vec.append(final_memory_usage - initial_memory_usage)
    
    df.unpersist()
    del df, pdf
    gc.collect()

spark.stop()
final_memory_usage = get_memory_usage()
ram_needed = final_memory_usage - initial_memory_usage

with open('/output/ram_time_values.csv', 'w') as file:
    file.write("Index,Execution Time (seconds),RAM Usage (B)\n")
    for i in range(100):
        file.write(f"{i+1},{time_vec[i]},{ram_vec[i]}\n")