#!/bin/bash		

# Setup for Kudu training

export KUDU=/home/training/training_materials/kudu
export KDATA=/home/training/training_materials/kudu/data

main() {
	echo "[setup.sh] Sarting Kudu course setup at $(date '+%Y-%m-%d %T')"

    # Define environment variables
    echo "export KUDU=$KUDU" >> ~/.bashrc
    echo "export KDATA=$KDATA" >> ~/.bashrc    

    setup_hdfs
    setup_hive_tables
	echo "[setup.sh] Kudu course setup complete $(date '+%Y-%m-%d %T')"
}

setup_hdfs() {
    hdfs dfs -mkdir -p /loudacre_kudu
    hdfs dfs  -chmod -R a+w /loudacre_kudu
}

setup_hive_tables() {
	echo "[setup.sh] Setting up example Impala/Hive tables"

    # Load devices data into default location for Hive tables
    
    # Devices
    hdfs dfs -mkdir -p /user/hive/warehouse/loudacre_kudu.db/devices_hive/
    hdfs dfs -put $KDATA/devices.csv /user/hive/warehouse/loudacre_kudu.db/devices_hive/
    
    # Accounts 
    hdfs dfs -mkdir -p /user/hive/warehouse/loudacre_kudu.db/accounts_hive/
    hdfs dfs -put $KDATA/accounts.parquet /user/hive/warehouse/loudacre_kudu.db/accounts_hive/

    # Device Status
    hdfs dfs -mkdir -p /user/hive/warehouse/loudacre_kudu.db/device_status_hive/
    hdfs dfs -put $KDATA/devicestatus.csv /user/hive/warehouse/loudacre_kudu.db/device_status_hive/
    
    # Set up the table to reference the data
    impala-shell -f $KUDU/scripts/kudu-hive-tables.sql

}


main
