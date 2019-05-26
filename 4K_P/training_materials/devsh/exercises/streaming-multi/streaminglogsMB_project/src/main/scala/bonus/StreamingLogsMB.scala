// Test script:
// python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
// Run application (multi host):
// spark2-submit --class bonus.StreamingLogsMB target/streamlogmb-1.0.jar gateway 1234
// Run application (single host):
// spark2-submit --master local[2] --class bonus.StreamingLogsMB target/streamlogmb-1.0.jar gateway 1234

package bonus

import org.apache.spark.SparkContext
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds

object StreamingLogsMB {
  
  // Given an array of new counts, add up the counts 
  // and then add them to the old counts and return the new total
  def updateCount = (newCounts: Seq[Int], state: Option[Int]) => {
     val newCount = newCounts.foldLeft(0)(_ + _)
     val previousCount = state.getOrElse(0)
     Some(newCount + previousCount) 
  }
  
  
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: bonus.StreamingLogsMB <hostname> <port>")
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
    // val countStream = logStream.countByWindow(Seconds(5),Seconds(2))
    // countStream.print()
    
    // Bonus: Display the top 5 users every second
    
    // Count requests by user ID for every batch
    val userreqStream = logStream.
       map(line => (line.split(' ')(2),1)).
       reduceByKey((x,y) => x+y)

    // Update total user requests
    val totalUserreqStream = userreqStream.updateStateByKey(updateCount)

    // Sort each state RDD by hit count in descending order    
    val topUserreqStream = totalUserreqStream.
       map(pair => pair.swap).
       transform(rdd => rdd.sortByKey(false)).
       map(pair => pair.swap)

    topUserreqStream.print
    


    ssc.start()
    ssc.awaitTermination()
  }

}