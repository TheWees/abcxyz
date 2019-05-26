#!/bin/bash

# This script will advance the state of the VM as if the
# exercise specified on the command line (and all those 
# before it) had been completed. For example, invoking 
# this script as:
#
#   $ advance_exercises.sh exercise4
#
# Will prepare you to begin work on exercise 5, meaning 
# that the state of the VM will be exactly the same as if 
# exercise 4 (as well as exercises 3, 2, and 1) had been
# manually completed.
#
# BEWARE: In all invocations, this script will first run
#         a 'cleanup' step which removes data in HDFS and
#         the local filesystem in order to simulate the
#         original state of the VM. 
#

DEVSH=/home/training/training_materials/devsh
DEVDATA=/home/training/training_materials/devsh/data

# ensure we run any scripts from a writable local directory, 
# which needs to also not conflict with anything
RUNDIR=/tmp/devsh/exercisescripts/$RANDOM$$
mkdir -p $RUNDIR
cd $RUNDIR

# Detect if this script is running in a single-host or multi-host environment
# TBD This should be implemented in a more elegant and maintainable way.
# See https://jira.cloudera.com/browse/CUR-5972
if [ $HOSTNAME == "localhost.localdomain" ]; then 
   SINGLEHOST=true
else 
   SINGLEHOST=false
fi

# Kafka settings 
if [ $SINGLEHOST ]; then
   ZOOKEEPER_SERVERS=master-1:2181
   KAFKA_REPLICATION=1
else
   ZOOKEEPER_SERVERS=master-1:2181,master-2:2181,worker-2:2181
   KAFKA_REPLICATION=3
fi

cleanup() {
    echo "Cleaning up your system"

    sudo -u hdfs hdfs dfs -rm  -skipTrash -r -f /loudacre

    kafka-topics --if-exists  --zookeeper $ZOOKEEPER_SERVERS  --delete --topic weblogs
}

exercise1() {
    echo "* Advancing through Exercise: Querying Hadoop Data with Apache Impala"
    # Nothing required for subsequent exercises
}

exercise2() {
    echo "* Advancing through Exercise: Accessing HDFS with the Command Line and Hue"
    
    # required for later exercises
    hdfs dfs -mkdir /loudacre
    hdfs dfs -put $DEVDATA/kb /loudacre/
    hdfs dfs -put $DEVDATA/base_stations.parquet /loudacre
}

exercise3() {
    # YARN
    echo "* Advancing through Exercise: Running and Monitoring a YARN Job"

    # Nothing required for subsequent exercises

}

exercise4() {
    echo "* Advancing through Exercise: Exploring DataFrames Using the Apache Spark Shell"
 
     # data required for this exercise and also for later exercises
    hdfs dfs -put $DEVDATA/devices.json /loudacre/

}


exercise5() {
    echo "* Advancing through Exercise: Working With DataFrames and Schemas"
    
    # Copy in data required for exercise (also required for later exercises)
    # No data written in this exercise

    # Copy in solution data
    hdfs dfs -put $DEVDATA/static_data/accounts_zip94913/ /loudacre/
    hdfs dfs -put $DEVDATA/static_data/devices_parquet/ /loudacre/
}

exercise6() {
    echo "* Advancing through Exercise: Analyzing Data with DataFrame Queries"
    
    # data required to complete this exercise (also required for later exercises)
    hdfs dfs -put $DEVDATA/accountdevice/ /loudacre/

    # Copy in solution data
    hdfs dfs -put $DEVDATA/static_data/top_devices/ /loudacre/

}

exercise7() {
    echo "* Advancing through Exercise: Working with RDDs"

    # data required to complete this exercise
    hdfs dfs -put $DEVDATA/frostroad.txt /loudacre/    
    hdfs dfs -put $DEVDATA/makes1.txt /loudacre/  
    hdfs dfs -put $DEVDATA/makes2.txt /loudacre/  

    # Solution data
    # No files saved in this exercise
}

exercise8() {
    echo "* Advancing through Exercise: Transforming Data Using RDDs"

	# This function sometimes causes HDFS warnings which can be disregarded
    # (to minimize this, we only upload data necessary for subsequent exercises)

    # data required to complete this exercise
    hdfs dfs -put $DEVDATA/weblogs/ /loudacre/  
    hdfs dfs -put $DEVDATA/activations/ /loudacre/  

    # Solution data
    #hdfs dfs -put $DEVDATA/static_data/iplist/ /loudacre/ 
    hdfs dfs -put $DEVDATA/static_data/devicestatus_etl /loudacre/
    #hdfs dfs -put $DEVDATA/static_data/userips_csv/ /loudacre/  
    #hdfs dfs -put $DEVDATA/static_data/account-models/ /loudacre/ 
}

exercise9() {
    echo "* Advancing through Exercise: Joining Data Using Pair RDDs"
    # No exercise output
}

exercise10() {
    echo "* Advancing through Exercise: Querying Tables and Views with SQL"

    # not required for later exercises, but students may want it later
    hdfs dfs -put $DEVDATA/static_data/name_dev/ /loudacre/  
}


exercise11() {
    echo "* Advancing through Exercise: Using Datasets in Scala"

    hdfs dfs -put $DEVDATA/static_data/accountIPs/ /loudacre/  
}

exercise12() {
    echo "* Advancing through Exercise: Writing, Configuring, and Running a Spark Application"
    
    # copying for student information, not required by subsequent exercises
    
    # Not copying accounts_by_state because content varies by state student chooses
    # hdfs dfs -put $DEVDATA/static-data/accounts_by_state /loudacre/
}

exercise13() {
    echo "* Advancing through Exercise: Exploring Query Execution"
    # No exercise output
}

exercise14() {
    echo "* Advancing through Exercise: Persisting Data"

    # not required for later exercises, but students may want it later
    hdfs dfs -put $DEVDATA/static_data/account_names_devices /loudacre
}

exercise15() {
    echo "* Advancing through Exercise: Implementing an Iterative Algorithm with Apache Spark"
    # No exercise output
}

exercise16() {
    echo "* Advancing through Exercise: Writing a Streaming Application"
    # Exercise output not required for later exercises
}

exercise17() {
    echo "* Advancing through Exercise: Processing Multiple Batches of Streaming Data"
    # Exercise output not required for later exercises
}

exercise18() {
    echo "* Advancing through Exercise: Processing Streaming Apache Kafka Messages"
    
    kafka-topics --create  --if-not-exists  --zookeeper $ZOOKEEPER_SERVERS  --replication-factor $KAFKA_REPLICATION  --partitions 2   --topic weblogs

    # Exercise output not required for later exercises
}

exercise19() {
    echo "* Advancing through Exercise: Producing and Consuming Apache Kafka Messages"

    kafka-topics --create  --if-not-exists  --zookeeper $ZOOKEEPER_SERVERS  --replication-factor 1 --partitions 2   --topic weblogs

}

exercise20() {
    # Flume
    echo "* Advancing through Exercise: Collecting Web Server Logs with Apache Flume"
    
    hdfs dfs -mkdir -p /loudacre/weblogs_flume
    hdfs dfs -put $DEVDATA/static_data/weblogs_flume /loudacre/
    
    sudo mkdir -p /flume/weblogs_spooldir
    sudo chmod a+w -R /flume

}

exercise21() {
    # Flafka
    echo "* Advancing through Exercise: Sending Messages from Flume to Kafka"
    # no exercise output
}



case "$1" in
        
    cleanup)
        cleanup
        ;;

    exercise1)
        cleanup
        exercise1
        ;;

    exercise2)
        cleanup
        exercise1
        exercise2
        ;;

    exercise3)
        cleanup
        exercise1
        exercise2
        exercise3
        ;;

   exercise4)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        ;;

        
    exercise5)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        ;;

   exercise6)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        ;;

    exercise7)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        ;;

    exercise8)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
       ;;

    exercise9)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        ;;

    exercise10)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        ;;
        
    exercise11)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        ;;
        
    exercise12)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        ;;

    exercise13)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        ;;

    exercise14)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        ;;
        
    exercise15)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        ;;

        
    exercise16)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        exercise16
        ;;
                

        
    exercise17)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        exercise16
        exercise17
        ;;
          
    exercise18)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        exercise16
        exercise17
        exercise18
        ;;
                 
           
    exercise19)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        exercise16
        exercise17
        exercise18
        exercise19
        ;;
                
             
    exercise20)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        exercise16
        exercise17
        exercise18
        exercise19
        exercise20
        ;;
                
           
    exercise21)
        cleanup
        exercise1
        exercise2
        exercise3
        exercise4
        exercise5
        exercise6
        exercise7
        exercise8
        exercise9
        exercise10
        exercise11
        exercise12
        exercise13
        exercise14
        exercise15
        exercise16
        exercise17
        exercise18
        exercise19
        exercise20
        exercise21
        ;;
                
            
    *)
        echo $"Usage: $0 {exercise1-exercise21}"
        exit 1

esac
