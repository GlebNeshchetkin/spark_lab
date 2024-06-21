# Spark Lab
<h3>Time and RAM Distribution</h3> 

![download](https://github.com/GlebNeshchetkin/spark_lab/assets/71218745/8f498e16-672a-4e0f-a3b7-2191587f1f5d)

<h3>Dataset</h3>
Car prices prediction dataset (car_prices_3.csv): year, make, model, trim, body, transmission, state, condition, odometer, color, interior, seller, mmr, sellingprice columns, 110000 rows.

<h3>Model</h3>
The catboost_model is the pretrained CatBoost regression model. Model training is in the Jupyter Notebook file in model_training folder.

<h3>Spark App</h3>
Two spark apps are created, for not parallel (simple_script_not_parallel.py) and parallel (simple_script_parallel.py) data processing.

<h3>Labels</h3>
labels.pkl file contains labels dictionary for data preprocessing.

<h3>How to Run</h3>
<h4>Load Data to HDFS</h4>

```sh
    docker-compose build
    docker-compose up -d
    docker cp car_prices_3.csv namenode:/
    docker exec -it namenode /bin/bash
```
```sh
    hdfs dfs -put car_prices_3.csv /
```

<h4>Start Spark App</h4>
The script start_script.sh retrieves the spark-master address from logs, transfers the spark app to the spark-master, and initiates its execution. Spark app name is specified in this script.

```sh
    ./start_script.sh
```
