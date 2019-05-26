#!/bin/bash
# For multi-host environments

# Create weblogs topic
kafka-topics --create   --zookeeper master-1:2181,master-2:2181,worker-2:2181   --replication-factor 3   --partitions 2   --topic weblogs

# List topics
kafka-topics --list    --zookeeper master-1:2181,master-2:2181,worker-2:2181

# Describe weblogs topic
kafka-topics --describe weblogs    --zookeeper master-1:2181,master-2:2181,worker-2:2181

