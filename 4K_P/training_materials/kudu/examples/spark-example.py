# Python examples Spark API

# Run $KUDU/scripts/create-tables.sh first
# pyspark2 --packages org.apache.kudu:kudu-spark2_2.11:1.5.0

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

kuduMaster = "master-2:7051"

customersDF = spark.read.format("org.apache.kudu.spark.kudu").option("kudu.master",kuduMaster).option("kudu.table","customers").load()

customersDF.printSchema()

# impala-shell -q "CREATE TABLE customers_CA (customer_id INT, name STRING, city STRING, PRIMARY KEY (customer_id)) PARTITION BY HASH PARTITIONS 2 STORED AS KUDU TBLPROPERTIES('kudu.table_name' = 'customers_CA')"

customersCADF = customersDF.where("state = 'CA'").select("customer_id","name","city")
customersCADF.printSchema()
customersCADF.write.format("org.apache.kudu.spark.kudu").option("kudu.master",kuduMaster).mode("append"). \
option("kudu.table","customers_CA").save()


# namesDF = spark.sql("SELECT customer_id,name FROM customers WHERE name LIKE 'B%'")
# error Caused by: java.lang.ClassNotFoundException: com.cloudera.kudu.hive.KuduStorageHandler

customersDF.createTempView("customers_view")
spark.sql("SELECT customer_id,name FROM customers_view WHERE name LIKE 'B%'").show()
