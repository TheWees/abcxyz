# Test script:
# python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
# Run solution application (multi host):
# spark2-submit solution-python/StreamingLogsMB.py gateway 1234
# Run solution application (single host):
# spark2-submit --master local[2] solution-python/StreamingLogsMB.py gateway 1234

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: StreamingLogsMB.py <hostname> <port>"
        sys.exit(-1)
    
    # get hostname and port of data source from application arguments
    hostname = sys.argv[1]
    port = int(sys.argv[2])
     
    # Create a new SparkContext
    sc = SparkContext()

    # Set log level to ERROR to avoid distracting extra output
    sc.setLogLevel("ERROR")

    # Create and configure a new Streaming Context 
    # with a 1 second batch duration
    ssc = StreamingContext(sc,1)

    # Enable checkpointing (required for window operations)
    ssc.checkpoint("logcheckpt")

    # Create a DStream of log data from the server and port specified    
    logStream = ssc.socketTextStream(hostname,port)

    # Every two seconds, display the total number of requests over the 
    # last 5 seconds
    countStream = logStream.countByWindow(5,2)

    # Example of simple output
    countStream.pprint()

    # Example of formatted output
    #def printRDDfirst(rdd,time):
    #  print "Number of requests @", time, " in last five seconds: ",rdd.first()
    #countStream.foreachRDD(lambda time,rdd: printRDDfirst(rdd,time))
   
    ssc.start()
    ssc.awaitTermination()
