
hdfs dfs -put $DEVSH/examples/example-data/people.json

# run Python application
spark2-submit --master yarn  --deploy-mode cluster $DEVSH/examples/spark/spark-applications/NameList.py people.json namelist/

spark2-submit --conf spark.pyspark.python=/usr/bin/python2.7

spark2-submit --properties-file=$DEVSH/examples/spark/spark-applications/my-properties.conf