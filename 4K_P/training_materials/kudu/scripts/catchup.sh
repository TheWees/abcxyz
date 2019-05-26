#!/bin/bash

# This catchup script does not follow the typical pattern of prompting the user
# for what exercise to advance to.  This is because only one exercise - the last one -
# depends on any prior exercise, so there is no reason to run this script except
# for that one

echo [Kudu] Running catchup script for Kudu

# Clean up before advancing

echo '* Dropping tables'
impala-shell --quiet -f $KUDU/scripts/cleanup-tables.sql

# Clean up external Kudu tables
kudu table delete master-2:7051 account_names_kudu

echo '* Advancing through Exercise: Exploring the Apache Kudu Cluster'
echo 'Nothing required'

echo '* Advancing through Exercise: Exploring Apache Kudu Tables'
# Subsequent chapters are not actually dependent on this but it is what happened in the exercise, so....
echo 'Creating example tables'
impala-shell --quiet -f $KUDU/scripts/example-tables.sql

echo '* Advancing through Exercise: Using Apache Kudu with Apache Impala'
impala-shell --quiet -f $KUDU/scripts/impala-catchup.sql

echo [Kudu] Done, you can start the exercise now
