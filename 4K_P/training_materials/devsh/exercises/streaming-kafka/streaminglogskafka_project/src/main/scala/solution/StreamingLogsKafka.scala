// Test script:
// $DEVSH/scripts/streamtest-kafka.sh weblogs worker-1:9092 20 $DEVDATA/weblogs/*
// Run application (multi host):
//spark2-submit --class solution.StreamingLogsKafka target/streamlogkafka-1.0.jar weblogs worker-1:9092
// Run application (single host):
//spark2-submit --master local[2] --class solution.StreamingLogsKafka target/streamlogkafka-1.0.jar weblogs worker-1:9092

package solution

import org.apache.spark.SparkContext
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.kafka._
import kafka.serializer.StringDecoder

object StreamingLogsKafka {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: solution.StreamingLogsKafka <topic> <brokerlist>")
      System.exit(1)
    }  
    val topic = args(0)
    val brokerlist = args(1)
    
    val sc = new SparkContext()
    sc.setLogLevel("ERROR")

    // Configure the Streaming Context with a 1 second batch duration
    val ssc = new StreamingContext(sc,Seconds(1))

    // Create a DStream of log data from Kafka topic  
    val kafkaStream = KafkaUtils.
       createDirectStream[String,String,StringDecoder,StringDecoder](ssc, 
       Map("metadata.broker.list"->brokerlist),
       Set(topic))

    // The weblog data is in the form (key, value), map to just the value
    val logStream = kafkaStream.map(pair => pair._2)

    // To test, print the first few lines of each batch of messages to confirm receipt
    logStream.print()
        
    // Print out the count of each batch RDD in the stream
    logStream.foreachRDD(rdd => println("Number of requests: " + rdd.count()))


    // Save the logs
    logStream.saveAsTextFiles("/loudacre/streamlog/kafkalogs")
    
    ssc.start()
    ssc.awaitTermination()
  }
}