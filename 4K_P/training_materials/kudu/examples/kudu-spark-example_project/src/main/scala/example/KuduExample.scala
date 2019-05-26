package example

import org.apache.spark.sql.SparkSession

object KuduExample {
  def main(args: Array[String]) {
     
    val spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val kuduMaster = "master-2:7051"

    val customersDF = spark.read.format("org.apache.kudu.spark.kudu").option("kudu.master",kuduMaster).option("kudu.table","customers").load()

    customersDF.printSchema

    val kc = new KuduContext(kuduMaster,spark.sparkContext)

    spark.stop()
  }
}

