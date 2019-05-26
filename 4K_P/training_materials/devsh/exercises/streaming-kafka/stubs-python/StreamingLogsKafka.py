# Test script:
# $DEVSH/scripts/streamtest-kafka.sh weblogs worker-1:9092 20 $DEVDATA/weblogs/*
# Run application (multi host):
# spark2-submit stubs-python/StreamingLogsKafka.py weblogs worker-1:9092
# Run application (single host):
# spark2-submit --master local[2] stubs-python/StreamingLogsKafka.py weblogs worker-1:9092

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

def printRDDcount(rdd): 
    print "Number of requests: "+str(rdd.count())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: StreamingLogsKafka.py <topic> <brokerlist>"
        sys.exit(-1)
    
    topic = sys.argv[1]
    brokerlist = sys.argv[2]
     
    sc = SparkContext()   
    sc.setLogLevel("ERROR")

    # Configure the Streaming Context with a 1 second batch duration
    ssc = StreamingContext(sc,1)
        
    # TODO
    print "Stub not yet implemented"
