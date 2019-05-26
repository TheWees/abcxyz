// Test script:
// python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
// Run application (multi host):
// spark2-submit --class solution.StreamingLogsMB target/streamlogmb-1.0.jar gateway 1234
// Run application (single host):
// spark2-submit --master local[2] --class solution.StreamingLogsMB target/streamlogmb-1.0.jar gateway 1234

package solution

import org.apache.spark.SparkContext
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds

object StreamingLogsMB {
  
 
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: solution.StreamingLogsMB <hostname> <port>")
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

    // Enable checkpointing (required for all window and state operations)
    ssc.checkpoint("logcheckpt")
    
    // Create a DStream of log data from the server and port specified   
    val logStream = ssc.socketTextStream(hostname,port)

    // Every two seconds, display the total number of requests over the 
    // last 5 seconds
    
    // Example of simple output
    val countStream = logStream.countByWindow(Seconds(5),Seconds(2))
    countStream.print

    // Example of formatted output
    // countStream.foreachRDD((rdd,time) => printf("Number of requests @ %s in last five seconds: %s\n",time,rdd.first))

    ssc.start()
    ssc.awaitTermination()
  }

}