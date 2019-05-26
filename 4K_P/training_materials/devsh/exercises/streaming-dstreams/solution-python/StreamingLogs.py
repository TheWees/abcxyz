# Test script:
# python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
# Run solution application (single host):
# spark2-submit --master local[2] solution-python/StreamingLogs.py gateway 1234
# Run solution application (multi host):
# spark2-submit solution-python/StreamingLogs.py gateway 1234

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Given an RDD of KB requests, print out the count of elements
def printRDDcount(rdd): print "Number of KB requests: "+str(rdd.count())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: StreamingLogs.py <hostname> <port>"
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

    # Create a DStream of log data from the server and port specified    
    logs = ssc.socketTextStream(hostname,port)

    # Filter the DStream to only include lines containing the string KBDOC
    kbreqStream = logs.filter(lambda line: "KBDOC" in line)

    # Test application by printing out the first 5 lines received in each batch 
    kbreqStream.pprint(5)

    # Print out the count of each batch RDD in the stream
    kbreqStream.foreachRDD(lambda t,r: printRDDcount(r))

    # Save the filtered logs
    kbreqStream.saveAsTextFiles("/loudacre/streamlog/kblogs")

    # Start the streaming context and then wait for application to terminate
    ssc.start()
    ssc.awaitTermination()
