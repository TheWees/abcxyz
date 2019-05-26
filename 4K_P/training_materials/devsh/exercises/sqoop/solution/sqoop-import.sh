sqoop import \
--connect jdbc:mysql://gateway/loudacre \
--username training --password training \
--table basestations \
--target-dir /loudacre/basestations_import


hdfs dfs -tail /loudacre/basestations_import/part-m-00001
hdfs dfs -tail /loudacre/basestations_import/part-m-00002
hdfs dfs -tail /loudacre/basestations_import/part-m-00003


sqoop import \
--connect jdbc:mysql://gateway/loudacre \
--username training --password training \
--table basestations \
--target-dir /loudacre/basestations_import_parquet \
--as-parquetfile

parquet-tools head  hdfs://master-1/loudacre/basestations_import_parquet
