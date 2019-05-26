import sys
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")


hostname = "localhost"
dbname = ""
jdbcPort = 3306
jdbc_url = "jdbc:mysql://{0}:{1}/{2}".format(hostname, jdbcPort, dbname)
connectionProperties = {
  "user" : "root" ,
  "password" : "" 
}
baseDF = spark.read.jdbc(url=jdbc_url, table="sales", properties=connectionProperties)
aggregatedByMonthDF=baseDF.groupBy(month(baseDF.orderDate).alias("orderMonth"),year(baseDF.orderDate).alias("orderYear")).agg(sum(baseDF.orderValue).alias('revenue'))
windowSpec = Window.partitionBy(aggregatedByMonthDF['orderYear']).orderBy(aggregatedByMonthDF['revenue'].desc())
rank= rank().over(windowSpec)
rankedDF = aggregatedByMonthDF.select(aggregatedByMonthDF['orderMonth'],aggregatedByMonthDF['orderYear'],aggregatedByMonthDF['revenue'],rank.alias("rank"))
topRevenueDF=rankedDF.where(rankedDF['rank'] == 1).orderBy(rankedDF['revenue'].desc())
outputDF=topRevenueDF.select(topRevenueDF['orderMonth'],topRevenueDF['orderYear'],round(topRevenueDF['revenue']/1000000,2).alias("Revenue in Millions"))
outputDF.show()
spark.stop()