DROP TABLE IF EXISTS customers_hash_id; 
CREATE TABLE customers_hash_id
PRIMARY KEY (customer_id)
PARTITION BY HASH(customer_id) PARTITIONS 4
STORED AS KUDU
AS SELECT * FROM customers;

--  hash by date
DROP TABLE IF EXISTS customers_hash_date; 
CREATE TABLE customers_hash_date
PRIMARY KEY (customer_id,created)
PARTITION BY HASH(created) PARTITIONS 4
STORED AS KUDU
AS SELECT * FROM customers;

--  range by date
DROP TABLE IF EXISTS customers_range_date; 
CREATE TABLE customers_range_date
PRIMARY KEY (customer_id,created)
PARTITION BY RANGE(created)
(
    PARTITION VALUE = 2013-01,
    PARTITION VALUE = 2013-02,
    PARTITION VALUE = 2013-03,
    PARTITION VALUE = 2013-04
)
STORED AS KUDU
AS SELECT * FROM customers;


ALTER TABLE customers_range_date ADD RANGE PARTITION VALUE = 2017-01;


--  Multi-level hash
DROP TABLE IF EXISTS customers_multi; 
CREATE TABLE customers_multi
PRIMARY KEY (customer_id,created)
PARTITION BY HASH(customer_id) PARTITIONS 4, RANGE(created)
(
    PARTITION VALUE = 2010,
    PARTITION VALUE = 2011,
    PARTITION VALUE = 2012
)
STORED AS KUDU
AS SELECT * FROM customers;

