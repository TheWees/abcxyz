-- # setup accounts table
DROP TABLE IF EXISTS accounts;

CREATE EXTERNAL TABLE accounts (
    acct_num INT,
    acct_create_dt TIMESTAMP,
    acct_close_dt  TIMESTAMP,
    first_name VARCHAR(255)  ,
    last_name VARCHAR(255) ,
    address  VARCHAR(255) ,
    city  VARCHAR(255) ,
    state VARCHAR(255) ,
    zipcode VARCHAR(255) ,
    phone_number VARCHAR(255) ,
    created TIMESTAMP  ,
    modified TIMESTAMP) 
     ROW FORMAT DELIMITED 
    FIELDS TERMINATED BY ','
;
