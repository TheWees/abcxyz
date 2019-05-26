# test using the streamtest.py script
# e.g. python $DEVSH/examples/spark-streaming/streamtest.py localhost 1234 $DEVDATA/static_data/weblogs/*

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: StreamingRequestCount.py <hostname> <port>"
        exit(-1)
        
    ssc = StreamingContext(SparkContext(),2)
    
    #hostname='localhost'
    hostname = sys.argv[1]
    #port=1234
    port = int(sys.argv[2])

    # ----- Example 1: Count the number of requests for each user in each batch ----- 

    # create a new dstream with (userid,numrequests) pairs
    # count the total number of requests from each userID for each batch.
    logs = ssc.socketTextStream(hostname, port)
    userreqs = logs \
      .map(lambda line: (line.split(' ')[2],1)) \
      .reduceByKey(lambda v1,v2: v1+v2)

    #userreqs.pprint()

    # save the full list of user/count pairs for each patch to HDFS files
    userreqs.saveAsTextFiles("streamreq/reqcounts")

    # ----- Example 2: Display the top 5 users in each batch ----- 

    # create a new dstream by transforming each rdd: 
    # map to swap key (userid) and value (numrequests, sort by numrequests in descending order
    sortedreqs=userreqs.map(lambda (k,v): (v,k)).transform(lambda rdd: rdd.sortByKey(False))


    def printTop5(r,t):
        print "Top users @",t
        for count,user in r.take(5):  print "User:",user,"("+str(count)+")"

    # print out the top 5 numrequests / userid pairs
    sortedreqs.foreachRDD(lambda time,rdd: printTop5(rdd,time))
    

    # ----- Example 3: total counts for all users over time ----- 
    # checkpointing must be enabled for state operations
    ssc.checkpoint("checkpoints")

    # Update function
    def updateCount(newCounts, state): 
       if state == None: return sum(newCounts)
       else: return state + sum(newCounts)
           
    totalUserreqs = userreqs.updateStateByKey(lambda newCounts, state: updateCount(newCounts, state))

    # Function to print state output
    def printTotalUsers(rdd):
        print "Total users:",rdd.count()
        for elem in rdd.take(5): print elem

    totalUserreqs.foreachRDD(lambda rdd: printTotalUsers(rdd))

    # ----- Example 4: Display top 5 users over 30 second window, output every 6 seconds  ----- 

    reqcountsByWindow = logs. \
        map(lambda line: (line.split(' ')[2],1)).\
        reduceByKeyAndWindow(lambda v1,v2: v1+v2, 30,6)
    topreqsByWindow=reqcountsByWindow. \
        map(lambda (k,v): (v,k)).transform(lambda rdd: rdd.sortByKey(False))

    def printTop5ByWindow(r):
        print "Top users by window"
        for count,user in r.take(5):  print "User:",user,"("+str(count)+")"

    # Function to print top users
    topreqsByWindow.foreachRDD(lambda rdd: printTop5ByWindow(rdd))

    ssc.start()
    ssc.awaitTermination()
    ssc.stop()

