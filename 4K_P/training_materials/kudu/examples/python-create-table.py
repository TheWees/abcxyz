# Example: Create a table in Python using kudu-python API

import kudu
from kudu.client import Partitioning

kuduMasterHost = 'master-2'
kuduMasterPort = 7051
kuduMaster = kuduMasterHost+':'+str(kuduMasterPort)

# Connect to Kudu master server
client = kudu.connect(host=kuduMasterHost, port=kuduMasterPort)

# Define a schema for a new table
builder = kudu.schema_builder()
builder.add_column('acct_num').type(kudu.int32).nullable(False).primary_key()
builder.add_column('first_name').type(kudu.string)
builder.add_column('last_name').type(kudu.string)
nameSchema = builder.build()
partitioning = Partitioning().add_hash_partitions(column_names=['acct_num'], num_buckets=2)
client.create_table('python-example', nameSchema, partitioning)

# confirm creation
for table in client.list_tables(): print table
print client.table("python-example").schema

# Test with Spark
namesDF = spark.read.table("accounts").select("acct_num","first_name","last_name")   
namesDF.write.format('org.apache.kudu.spark.kudu').option('kudu.master',kuduMaster).option('kudu.table','python-example').mode("append").save()


