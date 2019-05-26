// Test script:
// # $DEVSH/scripts/streamtest-kafka.sh weblogs worker-1:9092 20 $DEVDATA/weblogs/*
// Run application (multi host):
//spark2-submit --class stubs.StreamingLogsKafka target/streamlogkafka-1.0.jar weblogs worker-1:9092
// Run application (single host):
//spark2-submit --master local[2] --class stubs.StreamingLogsKafka target/streamlogkafka-1.0.jar weblogs worker-1:9092

package stubs

import org.apache.spark.SparkContext
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.kafka._
import kafka.serializer.StringDecoder

object StreamingLogsKafka {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: stubs.StreamingLogsKafka <topic> <brokerlist>")
      System.exit(1)
    }  
    val topic = args(0)
    val brokerlist = args(1)
    
    val sc = new SparkContext()
    sc.setLogLevel("ERROR")

    // Configure the Streaming Context with a 1 second batch duration
    val ssc = new StreamingContext(sc,Seconds(1))
    // TODO
    println("Stub not yet implemented")

    ssc.start()
    ssc.awaitTermination()
        
  }
}
