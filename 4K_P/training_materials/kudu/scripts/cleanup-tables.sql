-- Clean up all tables created in exercises

-- Chapter 3: Tables
DROP TABLE IF EXISTS  customers_kudu.customers;
DROP TABLE IF EXISTS  customers_kudu.customers_hash_id;
DROP TABLE IF EXISTS  customers_kudu.customers_hash_date;
DROP TABLE IF EXISTS  customers_kudu.customers_hash_id ;
DROP TABLE IF EXISTS  customers_kudu.customers_range_date ;
DROP TABLE IF EXISTS  customers_kudu.customers_multi ;

-- Chapter 4: Impala 
DROP TABLE IF EXISTS loudacre_kudu.names;
DROP TABLE IF EXISTS customers_kudu.names;
DROP TABLE IF EXISTS loudacre_kudu.accounts_kudu ;
DROP TABLE IF EXISTS loudacre_kudu.devices_composite;
DROP TABLE IF EXISTS loudacre_kudu.devices_range ;

-- Chapter 4: Spark
DROP TABLE IF EXISTS default.account_names_kudu;
DROP TABLE IF EXISTS loudacre_kudu.account_names_kudu;
DROP TABLE IF EXISTS customers_kudu.account_names_kudu;
