CREATE DATABASE IF NOT EXISTS loudacre_kudu;

DROP TABLE IF EXISTS loudacre_kudu.devices_hive;
CREATE EXTERNAL TABLE loudacre_kudu.devices_hive 
(
  devnum INT,
  released STRING,
  make STRING,
  model STRING,
  dev_type STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

DROP TABLE IF EXISTS loudacre_kudu.accounts_hive;
CREATE EXTERNAL TABLE loudacre_kudu.accounts_hive 
  LIKE PARQUET '/user/hive/warehouse/loudacre_kudu.db/accounts_hive/accounts.parquet' 
  STORED AS PARQUET;  

DROP TABLE IF EXISTS loudacre_kudu.device_status_hive;
CREATE EXTERNAL TABLE loudacre_kudu.device_status_hive 
(
  tstamp TIMESTAMP ,
  model STRING,
  dev_id STRING,
  power_remaining INT,
  gps STRING,
  bluetooth STRING,
  wifi STRING,
  lat FLOAT,
  lon FLOAT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
