#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkConf, SparkContext
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import * 
from dateutil.parser import parse

import math
import pickle
import time


# In[2]:


t0_all = time.time()


# In[3]:


conf = SparkConf()
conf = (conf.setMaster('local[*]')
        .set('spark.num.executors', '17')
        .set('spark.executor.cores', '5')
        .set('spark.executor.memory', '19g')
        .set('spark.driver.memory', '19g')
        .set('spark.network.timeout', '100001')
        .set('spark.executor.heartbeatInterval', '100000')
        .set('spark.rpc.askTimeout', '100000'))
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)


# In[4]:


# sc.getConf().getAll()
sc._conf.getAll()
# print(sc._conf.get('spark.driver.memory'))


# In[5]:


# For running in server only
t0 = time.time()
didi_df = spark.read.format("csv").option("header", "false").option("mode", "DROPMALFORMED")     .load("./DIDI_datasets/xian/gps_*")
t1 = time.time()
total = t1-t0
print("time taken {}".format(total))


# In[6]:


# # For running in local in replace of the above (of cos can run in server too if u just want to run 1 file)
# t0 = time.time()
# didi_df = spark.read.format("csv").option("header", "false").option("mode", "DROPMALFORMED") \
#     .load("test_sample3") #test_sample3
# t1 = time.time()
# total = t1-t0
# print("time taken {}".format(total))


# In[7]:


t0 = time.time()
didi_df = didi_df.toDF("driveID","orderID","timestamp","longitude","latitude")
didi_df.show(10)
didi_df = didi_df.select(didi_df.driveID,didi_df.orderID,
                        didi_df.timestamp.cast(DoubleType()),
                        didi_df.longitude.cast(DoubleType()),
                        didi_df.latitude.cast(DoubleType()))
t1 = time.time()
total = t1-t0
print("time taken {}".format(total))


# In[8]:


print(didi_df.dtypes)
didi_df.printSchema()
didi_df.show(10)


# In[9]:


# # u may skip this chunk cos count() will take very long to run, run only when needed
# t0 = time.time()
# print(didi_df.count())
# t1 = time.time()
# total = t1-t0
# print("time taken {}".format(total))


# In[10]:


# #u may skip this chunk cos count() will take very long to run, run only when needed
# t0 = time.time()
# didi_df_grouped = didi_df.groupBy("driveID")
# didi_df_grouped.count().show(10)
# t1 = time.time()
# total = t1-t0
# print("time taken {}".format(total))


# In[11]:


# # For running in server only
# t0 = time.time()
# didi_df.write.parquet("didi_31_files.parquet")
# t1 = time.time()
# total = t1-t0
# print("time taken {}".format(total))


# # Once you saved in the file in parquet, you only need to start from this line after executing the 1st 2 chunks

# In[12]:


# # For running in server only
# t0 = time.time()
# didi_new = sqlContext.read.parquet("didi_31_files.parquet")
# t1 = time.time()
# total = t1-t0
# print("time taken {}".format(total))


# In[13]:


didi_agg = didi_df.withColumn(
        'timestamp',
        from_unixtime(didi_df.timestamp, 'yyyy-MM-dd HH:mm:ss')
    )


# In[14]:


didi_agg = didi_agg.withColumn('dayOfWeek', date_format(didi_agg['timestamp'], 'u').cast(IntegerType()))    .withColumn('dayType', when((col("dayOfWeek")==0)|(col("dayOfWeek")==6) | 
                                ((col("timestamp") >= lit('2016-10-01')) & (col("timestamp") <= lit('2016-10-07'))), 
                                1).otherwise(0))
didi_agg.show(10)


# In[15]:


#this will take a long time, no need to run, just for check data
# didi_agg.groupBy("dayOfWeek").count().show()
# didi_agg.groupBy("dayType").count().show()
# didi_agg = didi_agg.drop("dayOfWeek", "orderID")
# didi_agg.printSchema()


# In[16]:


def getTimePartition(timestamp):
    dt = parse(timestamp)
    timePart = int(datetime.strftime(dt, '%H'))*60 + int(datetime.strftime(dt, '%M'))
    return timePart

getTimePartitionUdf = udf(lambda timestamp: getTimePartition(timestamp), IntegerType())
didi_agg_01 = didi_agg.withColumn("timePartition", getTimePartitionUdf("timestamp"))
didi_agg_01.show(10)
didi_agg_01.printSchema()


# In[17]:


latMin = 34.21012
latMax = 34.28021
lonMin = 108.91254
lonMax = 108.99848

# latMin = 34.21012 + 4*(0.07009/10)
# latMax = 34.21012 + 6*(0.07009/10)
# lonMin = 108.91254 + 4*(0.08594/10)
# lonMax = 108.91254 + 6*(0.08594/10)

def getLatPartition(lat):
    latPart = math.floor(((lat - latMin) / (latMax - latMin)) * 100)
    return latPart
def getLonPartition(lon):
    lonPart = math.floor(((lon - lonMin) / (lonMax - lonMin)) * 100)
    return lonPart

getLatPartitionUdf = udf(lambda latitude: getLatPartition(latitude), IntegerType())
getLonPartitionUdf = udf(lambda longitude: getLonPartition(longitude), IntegerType())
didi_agg_02 = didi_agg_01.withColumn("latPartition", getLatPartitionUdf("latitude"))    .withColumn("lonPartition", getLonPartitionUdf("longitude"))    .filter((col("latPartition")>=0) & (col("lonPartition")>=0))
didi_agg_02.show(10)


# In[18]:


didi_agg_02 = didi_agg_02.drop("longitude", "latitude")
didi_agg_02.show(10)
didi_agg_02.printSchema()


# In[19]:


t0 = time.time()

import datetime
from pyspark.sql.functions import year, month, dayofmonth
# #convert string timestamp to timestamp type             
# didi_agg_03 = didi_agg_02.withColumn('date', date_format(didi_agg_02['timestamp'], 'd').cast(IntegerType()))
didi_agg_03 = didi_agg_02.withColumn('timestamp_in_min', 
                                     round((col("timestamp").cast('timestamp').cast('long'))/60).cast('integer'))



#create window with partition over date and location
# w = (Window.partitionBy('latPartition','lonPartition')\
#          .orderBy(col("timestamp_simple")).rangeBetween(-900, 900))
w = (Window.partitionBy('latPartition','lonPartition')         .orderBy(col("timestamp_in_min")).rangeBetween(-15, 15)) # dynamic time partition
# w = (Window.partitionBy('timePartition','latPartition','lonPartition')) # static time partition

#use collect_set and size functions to perform countDistinct over a window
didi_agg_03 = didi_agg_03.withColumn('distinct_cars', size(collect_set("driveID").over(w)))

didi_agg_03.show(10)

t1 = time.time()
total = t1-t0
print("time taken {}".format(total))


# In[20]:


# didi_agg_03.groupBy("distinct_cars").count().orderBy(desc("count")).show()
# print(didi_agg_03.count())


# In[21]:


# view number of records per location per minute per minute
didi_agg_04_grouped = didi_agg_03.groupBy("timestamp_in_min", "latPartition", "lonPartition")    .agg(countDistinct('driveID').alias("distinct_cars_in_a_min")).orderBy(desc("distinct_cars_in_a_min"))
didi_agg_04_grouped.show(10)
didi_agg_04_grouped.printSchema()


# In[22]:


didi_agg_05_grouped = didi_agg_04_grouped.withColumn('density', (avg(col("distinct_cars_in_a_min"))).over(w))
didi_agg_05_grouped.orderBy(desc("density")).show(20)
didi_agg_05_grouped.printSchema()
# print(didi_agg_05_grouped.count())


# In[23]:


didi_agg_03.printSchema()
didi_agg_05_grouped.printSchema()


# In[24]:


didi_agg_06 = didi_agg_03.join(didi_agg_05_grouped, 
                 [didi_agg_03.timestamp_in_min == didi_agg_05_grouped.timestamp_in_min,
                 didi_agg_03.latPartition == didi_agg_05_grouped.latPartition,
                 didi_agg_03.lonPartition == didi_agg_05_grouped.lonPartition])\
        .select("driveID", "dayType",didi_agg_03.timestamp_in_min,"timePartition",
               didi_agg_03.latPartition, didi_agg_03.lonPartition, "distinct_cars", "density")\
didi_agg_06.orderBy(desc("density")).show(10)
didi_agg_06.printSchema()


# In[25]:


didi_agg_07 = didi_agg_06.withColumn("traffic_flow_index", col("distinct_cars")/col("density"))                    .drop("distinct_cars", "density")
didi_agg_07.orderBy(desc("traffic_flow_index")).show(10)
didi_agg_07.printSchema()


# In[26]:


# see which block has the best / worst traffic flow
didi_agg_08_grouped = didi_agg_07.groupBy('latPartition','lonPartition')        .agg(avg("traffic_flow_index").alias("avg_traffic_flow_index"))
didi_agg_08_grouped.orderBy(desc("avg_traffic_flow_index")).show()
didi_agg_08_grouped.orderBy("avg_traffic_flow_index").show()


# In[27]:


# data cleaned for machine learning
didi_final = didi_agg_07
didi_final.show(10)


# In[29]:


# For running in server only
t0 = time.time()
didi_final.write.parquet("didi_31_files.parquet")
t1 = time.time()
total = t1-t0
print("time taken {}".format(total))


# # Simple random forest model (default setting)

# In[ ]:


# For running in server only
t0 = time.time()
didi_final = sqlContext.read.parquet("didi_31_files.parquet")
t1 = time.time()
total = t1-t0
print("time taken {}".format(total))


# In[ ]:


t0_rf = time.time()


# In[30]:


from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd


# In[31]:


from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
#create a rank for time, for the convience of selecting training, validation and test data
didi_df = didi_final.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("timestamp_in_min")))
didi_df.show(10)


# In[32]:


#use the rank column to select the data for training, validation and test
training_df = didi_df.where("rank <= 0.8").drop("rank")
training_df.show(3)

validation_df = didi_df.where("rank > 0.8").where("rank <= 0.9").drop("rank")
validation_df.show(3)

test_df = didi_df.where("rank > 0.9").where("rank <= 1.0").drop("rank")
test_df.show(3)


# In[33]:


features = ['dayType','timePartition','latPartition','lonPartition']  
training = training_df.select(col("traffic_flow_index"), *features)
training.printSchema()  

validation = validation_df.select(col("traffic_flow_index"), *features)  
validation.printSchema()  

test = test_df.select(col("traffic_flow_index"), *features) 
test.printSchema() 


# In[34]:


vector = VectorAssembler(inputCols=features, outputCol="features")
vtraining = vector.transform(training)
vtraining_df = vtraining.select(['features', 'traffic_flow_index'])
vtraining_df.show(3)

vvalidation = vector.transform(validation)
vvalidation_df = vvalidation.select(['features', 'traffic_flow_index'])
vvalidation_df.show(3)

vtest = vector.transform(test)
vtest_df = vtest.select(['features', 'traffic_flow_index'])
vtest_df.show(3)


# In[35]:


rfr = RandomForestRegressor(labelCol='traffic_flow_index')
rfr_model = rfr.fit(vtraining_df)


# In[36]:


#for validation data
rfr_predictions = rfr_model.transform(vvalidation_df)
rfr_predictions.select("prediction","traffic_flow_index","features").show(5)
rfr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="traffic_flow_index",metricName="rmse")
print("Root Mean Squared Error (RMSE) on validation data = %g" % rfr_evaluator.evaluate(rfr_predictions))


# In[37]:


# for test data, run after load all data
rfr_predictions = rfr_model.transform(vtest_df)
rfr_predictions.select("prediction","traffic_flow_index","features").show(5)
rfr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="traffic_flow_index",metricName="rmse")
print("Root Mean Squared Error (RMSE) on test data = %g" % rfr_evaluator.evaluate(rfr_predictions))


# In[58]:


t1_rf = time.time()
total = t1_rf-t0_rf
print("time taken {}".format(total))


# # Linear Regression

# In[ ]:


# Linear Regression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

features = ['dayType','timePartition','latPartition','lonPartition']  
didi = didi_final.select(col("traffic_flow_index"), *features)  
didi.printSchema()  

splits = vdidi_df.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]

lr = LinearRegression(featuresCol = 'features', labelCol='traffic_flow_index', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
# Print the coefficients and intercept for linear regression
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)


# In[ ]:


lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","traffic_flow_index","features").show(5) 
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# # Decision Tree Regression

# In[ ]:


#Decision Tree Regression
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'traffic_flow_index')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_predictions.select('prediction', 'traffic_flow_index', 'features').show(5)
dt_evaluator = RegressionEvaluator(
    labelCol="MV", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# # Gradient Boosted Trees

# In[ ]:


# Gradient Boosted Trees
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'traffic_flow_index', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'traffic_flow_index', 'features').show(5)

gbt_evaluator = RegressionEvaluator(
    labelCol="MV", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# # Output to csv

# In[49]:


# t0 = time.time()
# didi_final.toPandas().to_csv('didi_final_20161028.csv')
# t1 = time.time()
# total = t1-t0
# print("time taken {}".format(total))


# In[50]:


t0 = time.time()
rfr_predictions.toPandas().to_csv('rfr_predictions.csv')
t1 = time.time()
total = t1-t0
print("time taken {}".format(total))


# ## Get street name of the jam roads

# In[1]:


latMin = 34.21012
latMax = 34.28021
lonMin = 108.91254
lonMax = 108.99848

latPartitionWidth = (latMax - latMin)/100
lonPartitionWidth = (lonMax - lonMin)/100

def getLatFromPartition(latPartition):
    lat = latMin + latPartition*latPartitionWidth + latPartitionWidth/2
    return lat
def getLonFromPartition(lonPartition):
    lon = lonMin + lonPartition*lonPartitionWidth + lonPartitionWidth/2
    return lon


# In[2]:


print(getLatFromPartition(16))
print(getLatFromPartition(10))
print(getLatFromPartition(16))
print(getLonFromPartition(51))
print(getLonFromPartition(85))
print(getLonFromPartition(76))


# In[3]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="didi_taxi")
location = geolocator.reverse("34.22168485, 108.9567991")
print(location.address)


# In[4]:


#青龙路, 大雁塔, 雁塔区 (Yanta), 西安市, 陕西省, 710055, 中国
#慈恩东路, 曲江, 大雁塔, 雁塔区 (Yanta), 西安市, 陕西省, 710061, 中国
#曲江大道, 曲江街办, 雁塔区 (Yanta), 西安市, 陕西省, 710055, 中国
#东郡, 曲江街办, 雁塔区 (Yanta), 西安市, 陕西省, 中国
#miko cafe, 雁塔西路（东段）, 曲江, 小寨路, 雁塔区 (Yanta), 西安市, 陕西省, 710061, 中国

