# Spark Lab
<h3>Time and RAM Distribution</h3> 

![download](https://github.com/GlebNeshchetkin/spark_lab/assets/71218745/8f498e16-672a-4e0f-a3b7-2191587f1f5d)

<h3>Dataset</h3>
Car prices dataset is in data folder (car_prices_3.csv): year, make, model, trim, body, transmission, state, condition, odometer, color, interior, seller, mmr, sellingprice columns.

<h3>Model</h3>
The model folder contains the pretrained CatBoost regression model file. Model training is in the Jupyter Notebook file in model_training folder.

<h3>Spark App</h3>
Two spark apps are in spark folder, for not parallel (simple_script_not_parallel.py) and parallel (simple_script_parallel.py) data processing.

<h3>How to Run</h3>
<h4>Load Data to HDFS</h4>

```sh
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
