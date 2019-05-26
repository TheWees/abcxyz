#!/bin/bash

# Create Kafka topic for streaming weblogs
kafka-topics --create --zookeeper master-1:2181 --replication-factor 1 --partitions 2 --topic weblogs

kafka-topics --list    --zookeeper master-1:2181

kafka-topics --describe weblogs    --zookeeper master-1:2181

