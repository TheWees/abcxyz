-- Impala exercise solution

-- Work with example table: names_kudu

DROP TABLE IF EXISTS loudacre_kudu.names;
CREATE TABLE loudacre_kudu.names
(
  name STRING,
  age INTEGER,
  PRIMARY KEY(name)
)
PARTITION BY HASH PARTITIONS 2
STORED AS KUDU;

INSERT INTO loudacre_kudu.names VALUES 
  ('Grace Hopper',52),
  ('Ada Lovelace',28),
  ('Alan Turing',32);
  
INSERT INTO loudacre_kudu.names VALUES ('Grace Hopper',63);

UPSERT INTO loudacre_kudu.names VALUES 
  ('Grace Hopper',63),
  ('Katherine Johnson',99);
  
DROP TABLE loudacre_kudu.names;

-- Create a table based on existing Impala table
  
DROP TABLE IF EXISTS loudacre_kudu.accounts_kudu; 
CREATE TABLE loudacre_kudu.accounts_kudu 
  PRIMARY KEY (acct_num)
  PARTITION BY HASH PARTITIONS 2
  STORED AS KUDU
  AS SELECT * FROM loudacre_kudu.accounts_hive;

-- Create a table with a composite key

DROP TABLE IF EXISTS loudacre_kudu.devices_composite;
CREATE TABLE loudacre_kudu.devices_composite 
  PRIMARY KEY (make,model)
  PARTITION BY HASH (model) PARTITIONS 2
  STORED AS KUDU
  AS SELECT make,model,devnum,released,dev_type 
    FROM loudacre_kudu.devices_hive;

-- Create a table partitioned by date range
DROP TABLE IF EXISTS loudacre_kudu.devices_range; 
CREATE TABLE loudacre_kudu.devices_range 
  PRIMARY KEY (devnum,rel_year)
  PARTITION BY RANGE(rel_year) (
  	PARTITION VALUES < 2016,
  	PARTITION 2016 <= VALUES < 2017,
  	PARTITION 2017 <= VALUES < 2018
  )  
  STORED AS KUDU
  AS SELECT devnum,YEAR(released) as rel_year,released,make,model,dev_type 
    FROM loudacre_kudu.devices_hive;
    
    
  ALTER TABLE loudacre_kudu.devices_range 
    ADD RANGE PARTITION 2018 <= VALUES < 2019;