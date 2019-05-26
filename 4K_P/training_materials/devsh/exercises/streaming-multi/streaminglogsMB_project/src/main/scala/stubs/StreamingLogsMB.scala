// Test script:
// python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
// Run application (multi host):
// spark2-submit --class stubs.StreamingLogsMB target/streamlogmb-1.0.jar gateway 1234
// Run application (single host):
// spark2-submit --master local[2] --class stubs.StreamingLogsMB target/streamlogmb-1.0.jar gateway 1234

package stubs

import org.apache.spark.SparkContext
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds

object StreamingLogsMB {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: stubs.StreamingLogsMB <hostname> <port>")
      System.exit(1)
    } 
 
    // get hostname and port of data source from application arguments
    val hostname = args(0)
    val port = args(1).toInt

    // Create a Spark Context
    val sc = new SparkContext()

    // Set log level to ERROR to avoid distracting extra output
    sc.setLogLevel("ERROR")

    // Configure the Streaming Context with a 1 second batch duration
    val ssc = new StreamingContext(sc,Seconds(1))

    // Create a DStream of log data from the server and port specified   
    val logs = ssc.socketTextStream(hostname,port)



    // TODO
    println("Stub not yet implemented")



    ssc.start()
    ssc.awaitTermination()
  }
}