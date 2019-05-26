#!/bin/sh

flume-ng agent --conf /etc/flume-ng/conf --conf-file $DEVSH/exercises/flafka/spooldir_kafka.conf --name agent1 -Dflume.root.logger=INFO,console