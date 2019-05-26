-- Chapter example code

--  Create an empty table
DROP TABLE IF EXISTS loudacre_kudu.devices; 
CREATE TABLE loudacre_kudu.devices 
(
  devnum INTEGER,
  released STRING,
  make STRING,
  model STRING,
  dev_type STRING,
  PRIMARY KEY(devnum)
)
PARTITION BY HASH(devnum) PARTITIONS 3
STORED AS KUDU;

--  Create table with a different name and replication table property
DROP TABLE IF EXISTS loudacre_kudu.devices_kudu; 
CREATE TABLE loudacre_kudu.devices_kudu
(
  devnum INTEGER,
  released STRING,
  make STRING,
  model STRING,
  dev_type STRING,
  PRIMARY KEY(devnum)
)
PARTITION BY HASH(devnum) PARTITIONS 3
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'devices',
  'kudu.num_tablet_replicas' = '1'
);

--  Create a table based on an existing impala table 
-- (Kudu table must exist before running this command)
DROP TABLE IF EXISTS  loudacre_kudu.devices_external;
CREATE EXTERNAL TABLE loudacre_kudu.devices_external
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'devices_spark'
);

-- Create a table based on an existing impala table
-- CTAS
DROP TABLE IF EXISTS  loudacre_kudu.devices_ctas;
CREATE TABLE loudacre_kudu.devices_ctas
  PRIMARY KEY (devnum)
  PARTITION BY HASH(devnum) PARTITIONS 3
  STORED AS KUDU
  AS SELECT devnum,make,model FROM loudacre_kudu.devices_hive; 

-- Insert Rows
INSERT INTO loudacre_kudu.devices VALUES 
  (140,'2010-05','Titanic','2200','phone');
INSERT INTO loudacre_kudu.devices VALUES 
  (150,'2012-08','iFruit','4','phone'),
  (151,'2012-08','iFruit','4t','tablet');
INSERT INTO loudacre_kudu.devices VALUES 
 (150,'2017-01','Titanic','DeckChairs Plus','phone');
 
-- modify a single row based on primary key
UPDATE loudacre_kudu.devices SET model='5a' WHERE devnum = 150;
-- modify all rows returned by a query
UPDATE loudacre_kudu.devices SET model='5a' WHERE make='iFruit' AND model = '5';
UPSERT INTO loudacre_kudu.devices VALUES(152,'2017-01','Titanic','DeckChairs Minus','phone');

DELETE FROM loudacre_kudu.devices WHERE model = 'iFruit';


-- Hash Partitioning

CREATE TABLE loudacre_kudu.devices
( 
  devnum INTEGER,
  released STRING,
  make STRING,
  model STRING,
  dev_type STRING,
  PRIMARY KEY(devnum)
)
PARTITION BY HASH(devnum) PARTITIONS 3 STORED AS KUDU;

-- Range Partitioning by date

DROP TABLE IF EXISTS loudacre_kudu.devices_range; 
CREATE TABLE loudacre_kudu.devices_range 
(   
 devnum INTEGER,
  released STRING,
  make STRING,
  model STRING,
  dev_type STRING,
 PRIMARY KEY(devnum,released)
)
PARTITION BY RANGE(released) (
  PARTITION '2010-01' <= VALUES < '2017-01', 
  PARTITION VALUE = '2017-01',
  PARTITION VALUE = '2017-02',
  PARTITION VALUE = '2017-03')
STORED AS KUDU;


  -- Multilevel: Range Partitioning by date with hash partitioning by device number

DROP TABLE IF EXISTS loudacre_kudu.devices_multi; 
CREATE TABLE loudacre_kudu.devices_multi (   
 devnum INTEGER,
  released STRING,
  make STRING,
  model STRING,
  dev_type STRING,
 PRIMARY KEY(devnum,released)
)
PARTITION BY HASH(devnum) PARTITIONS 3, RANGE(released) (
  PARTITION '2010-01' <= VALUES < '2017-01',
  PARTITION VALUE = '2017-01',
  PARTITION VALUE = '2017-02',
  PARTITION VALUE = '2017-03' )
STORED AS KUDU;
   
-- Add and delete a range partition
ALTER TABLE loudacre_kudu.devices_range ADD  RANGE PARTITION VALUE = '2017-04';
ALTER TABLE loudacre_kudu.devices_range DROP RANGE PARTITION VALUE = '2017-01';
  
-- Quiz question example
DROP DATABASE IF EXISTS policy;
CREATE DATABASE policy;
CREATE TABLE policy.vehicles 
( 
   year INTEGER,
   make STRING,
   model STRING,
   trimline STRING,
   color STRING,
   PRIMARY KEY(year,make,model)
) 
PARTITION BY HASH(make) PARTITIONS 2
STORED AS KUDU;	

INSERT INTO policy.vehicles VALUES (2015, 'Honda', 'Accord', 'EX', 'Ruby Red');
INSERT INTO policy.vehicles VALUES (2015, 'Honda', 'Accord', 'EL', 'Obsidian Blue');
INSERT INTO policy.vehicles VALUES (2015, 'Honda', 'Civic', 'Si', 'Morning Shadow');