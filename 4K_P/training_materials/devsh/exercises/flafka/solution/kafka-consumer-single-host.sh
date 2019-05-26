#!/bin/sh

# For single-host environments
kafka-console-consumer   --zookeeper master-1:2181   --topic weblogs
