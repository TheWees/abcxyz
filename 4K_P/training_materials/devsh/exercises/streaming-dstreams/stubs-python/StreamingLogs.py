# Test script:
# python $DEVSH/scripts/streamtest.py gateway 1234 20 $DEVDATA/weblogs/*
# Run solution application (multi host):
# spark2-submit stubs-python/StreamingLogs.py gateway 1234
# Run solution application (single host):
# spark2-submit --master local[2] stubs-python/StreamingLogs.py gateway 1234

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

    # Set log level to WARN to avoid distracting extra output
    sc.setLogLevel("WARN")

    # TODO
    print "Stub not yet implemented"