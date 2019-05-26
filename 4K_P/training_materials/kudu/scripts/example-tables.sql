CREATE DATABASE IF NOT EXISTS customers_kudu;

DROP TABLE IF EXISTS customers_kudu.customers;
CREATE TABLE customers_kudu.customers
PRIMARY KEY (customer_id)
PARTITION BY HASH PARTITIONS 4
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'customers'
)
AS SELECT acct_num  AS customer_id, from_unixtime(acct_create_dt,'yyyy-MM') AS created, last_name AS name, city, state 
  FROM loudacre_kudu.accounts_hive;

-- hash by id
DROP TABLE IF EXISTS  customers_kudu.customers_hash_id; 
CREATE TABLE  customers_kudu.customers_hash_id
PRIMARY KEY (customer_id)
PARTITION BY HASH(customer_id) PARTITIONS 4
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'customers_hash_id'
)
AS SELECT * FROM  customers_kudu.customers;

--  hash by year/month
DROP TABLE IF EXISTS  customers_kudu.customers_hash_date; 
CREATE TABLE  customers_kudu.customers_hash_date
PRIMARY KEY (customer_id,created)
PARTITION BY HASH(created) PARTITIONS 4
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'customers_hash_date'
)
AS SELECT * FROM  customers_kudu.customers;

--  range by year/month
DROP TABLE IF EXISTS  customers_kudu.customers_range_date; 
CREATE TABLE  customers_kudu.customers_range_date
PRIMARY KEY (customer_id,created)
PARTITION BY RANGE(created)
(
    PARTITION VALUE = '2010-01',
    PARTITION VALUE = '2010-02',
    PARTITION VALUE = '2010-03'
)
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'customers_range_date'
)
AS SELECT * FROM  customers_kudu.customers;

-- Add a new year/month range as necessary
-- ALTER TABLE customers_range_date ADD RANGE PARTITION VALUE = 2012-05;


--  Multi-level hash
DROP TABLE IF EXISTS  customers_kudu.customers_multi; 
CREATE TABLE customers_kudu.customers_multi
PRIMARY KEY (customer_id,created)
PARTITION BY HASH(customer_id) PARTITIONS 4, RANGE(created)
(
    PARTITION VALUE = '2010-01',
    PARTITION VALUE = '2010-02',
    PARTITION VALUE = '2010-03'
)
STORED AS KUDU
TBLPROPERTIES(
  'kudu.table_name' = 'customers_multi'
)
AS SELECT * FROM  customers_kudu.customers;


