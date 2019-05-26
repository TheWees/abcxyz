# Copy sample data to a staging directory on HDFS for testing
# $ hdfs dfs -mkdir -p structstream_test
# $ hdfs dfs -put $DEVSH/examples/example-data/structstream_test/* structstream_test/

# Create test data directory 
# $ hdfs dfs -mkdir -p people

# Start application (if running locally add --master local[2])
#  spark2-submit StructStreamingExample.py 

# Move the file(s) from staging directory one by one into the directory people 
# $ hdfs dfs -mv structstream_test/part0.csv people/
# $ hdfs dfs -mv structstream_test/part1.csv people/
#  etc

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")
 
from pyspark.sql.types import *

columnsList = [
  StructField("lastName", StringType()),
  StructField("firstName", StringType()),
  StructField("pcode", StringType())
]

peopleSchema = StructType(columnsList)

peopleDF = spark.readStream.schema(peopleSchema).csv("people")

pcodeCountsDF = peopleDF.groupBy("pcode").count()

countsQuery = pcodeCountsDF.writeStream.outputMode("complete").format("console").start()

countsQuery.awaitTermination()
