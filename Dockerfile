FROM bitnami/spark:latest

USER root
RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install streamlit
RUN pip3 install catboost
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install psutil

USER 1001

COPY ./ram_time_values.csv /output/ram_time_values.csv

CMD ["bin/spark-class", "org.apache.spark.deploy.master.Master"]
