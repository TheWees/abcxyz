#!/bin/bash

# For single-host environments

# Start console producer
kafka-console-producer   --broker-list worker-1:9092 --topic weblogs

# Sample test data, paste into terminal window:
# test weblog entry 1

# Sample test data after consumer restart, paste into terminal window:
# test weblog entry 1