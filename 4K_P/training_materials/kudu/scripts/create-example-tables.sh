#!/bin/bash		

echo "[create-tables.sh] Creating example tables in Kudu"

impala-shell --quiet -f $KUDU/scripts/example-tables.sql

echo "[create-tables.sh] Example tables created"