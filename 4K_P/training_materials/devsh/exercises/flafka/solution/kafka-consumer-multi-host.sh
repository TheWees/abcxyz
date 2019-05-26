#!/bin/sh

# For multi-host environments

# Start the console consumer
kafka-console-consumer   --zookeeper master-1:2181,master-2:2181,worker-2:2181   --topic weblogs
