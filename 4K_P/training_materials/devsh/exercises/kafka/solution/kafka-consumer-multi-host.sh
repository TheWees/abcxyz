#!/bin/bash

# For multi-host environments

# Start the console consumer, receive all messages from the beginning of the topic
kafka-console-consumer   --zookeeper master-1:2181,master-2:2181,worker-2:2181   --topic weblogs   --from-beginning

# Re-start the console consumer, receive only new messages
kafka-console-consumer   --zookeeper master-1:2181,master-2:2181,worker-2:2181   --topic weblogs
  