#!/bin/bash

# For multi-host environments

# Start console producer
kafka-console-producer   --broker-list worker-1:9092,worker-2:9092,worker-3:9092   --topic weblogs

# Sample test data, paste into terminal window:
# test weblog entry 1

# Sample test data after consumer restart, paste into terminal window:
# test weblog entry 1