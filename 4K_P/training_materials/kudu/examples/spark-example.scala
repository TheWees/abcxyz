// Scala examples Spark API

// spark2-shell --packages org.apache.kudu:kudu-spark2_2.11:1.5.0

// Run $KUDU/scripts/create-tables.sh first

val kuduMaster = "master-2:7051"

val customersDF = spark.read.format("org.apache.kudu.spark.kudu").option("kudu.master",kuduMaster).option("kudu.table","customers").load()

customersDF.printSchema

// impala-shell -q "CREATE TABLE customers_CA (customer_id INT, name STRING, city STRING, PRIMARY KEY (customer_id)) PARTITION BY HASH PARTITIONS 2 STORED AS KUDU TBLPROPERTIES('kudu.table_name' = 'customers_CA')"

val customersCADF = customersDF.where("state = 'CA'").select("customer_id","name","city")
customersCADF.printSchema
customersCADF.write.format("org.apache.kudu.spark.kudu").option("kudu.master",kuduMaster).mode("append"). 
option("kudu.table","customers_CA").save

// val namesDF = spark.sql("SELECT customer_id,name FROM customers WHERE name LIKE 'B%'")
// error Caused by: java.lang.ClassNotFoundException: com.cloudera.kudu.hive.KuduStorageHandler

customersDF.createTempView("customers")
val namesDF = spark.sql("SELECT customer_id,name FROM customers WHERE name LIKE 'B%'")
namesDF.show(5)

// Managing tables
import org.apache.kudu.spark.kudu._
import org.apache.kudu.client._
import collection.JavaConverters._

val kc = new KuduContext(kuduMaster,sc)

import org.apache.spark.sql.types._

val namesColumns = List(
  StructField("customer_id",IntegerType),
  StructField("name",StringType))

val namesSchema = StructType(namesColumns)

// or
val namesDF = customersDF.select("customer_id","name")
val namesSchema = namesDF.schema

val namesTableOptions = new CreateTableOptions().addHashPartitions(List("customer_id").asJava, 2)
val namesTable = kc.createTable("names", namesSchema, List("customer_id"), namesTableOptions)

// test the new table
val namesDF2 = spark.read.format("org.apache.kudu.spark.kudu").option("kudu.master",kuduMaster).option("kudu.table","names").load()


// Delete data
val customersNVDF = customersDF.where("state = 'NV'").select("customer_id")
kc.deleteRows(customersNVDF, "customers")

// Upsert data
val newCustomers = spark.createDataFrame(List((4,"2017-12","test1","test2","MA"),(10000,"2017-13","new1","new2","MA"))).withColumnRenamed("_2","created").withColumnRenamed("_1","customer_id").withColumnRenamed("_3","name").withColumnRenamed("_4","city").withColumnRenamed("_5","state")
 kc.upsertRows(newCustomers, "customers")
customersDF.where("customer_id = 10000").show
customersDF.where("customer_id = 4").show