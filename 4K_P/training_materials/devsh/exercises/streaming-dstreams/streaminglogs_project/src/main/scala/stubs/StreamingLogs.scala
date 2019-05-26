// Test script:
// python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
// Run application (multi node):
// spark2-submit --class stubs.StreamingLogs target/streamlog-1.0.jar gateway 1234
// Run application (single node):
// spark2-submit --master local[2] --class stubs.StreamingLogs target/streamlog-1.0.jar gateway 1234

package stubs

import org.apache.spark.SparkContext
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds

object StreamingLogs {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: stubs.StreamingLogs <hostname> <port>")
      System.exit(1)
    } 
 
    // get hostname and port of data source from application arguments
    val hostname = args(0)
    val port = args(1).toInt
    
    // Create a new SparkContext
    val sc = new SparkContext()

    // Set log level to ERROR to avoid distracting extra output
    sc.setLogLevel("ERROR")
            
    // TODO
    println("Stub not yet implemented")

  }
}
