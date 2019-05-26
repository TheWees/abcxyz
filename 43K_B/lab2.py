#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import numpy as np 
from numpy import arange,array,ones
from scipy import stats 
from types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
import matplotlib.pyplot as plt  
import pandas as pd  
from pyspark.ml.fpm import FPGrowth
from collections import Counter
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import requests
requests.packages.urllib3.disable_warnings()
plotly.tools.set_credentials_file(
        username='weeliyen', api_key='uwNCB478l9TjFiGdVsaQ')

spark = SparkSession.builder.appName("Relevant Docs").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.setLogLevel("WARN")

### Read in metadata ###
if len(sys.argv) >=2 : 
    rawMetaDf = sqlContext.read.json(sys.argv[1])
else :
    rawMetaDf = sqlContext.read.json(os.getcwd() + "/meta_Grocery_and_Gourmet_Food.json")

### Read in reviews ###
if len(sys.argv) >=3 : 
    rawReviewDf = sqlContext.read.json(sys.argv[2])
else :
    rawReviewDf = sqlContext.read.json(os.getcwd() + "/reviews_Grocery_and_Gourmet_Food.json")

metaRelatedDf = rawMetaDf.select("asin","related").na.drop(subset=["asin"])
#metaRelatedDf.show()
#metaRelatedDf.count() #159637

metaRelatedDf_alsoBought = metaRelatedDf.select(
        "asin", col("related.also_bought").alias("also_bought"))
#metaRelatedDf_alsoBought.show()

############################################################
##
## 1. Frequently browsed together by the customers
##
############################################################

#===========================================================
## This method is to confirm that there is duplicates
##     Need to remove cos FPGrowth cannot accept duplicates
#===========================================================
#def checkForDuplicates(asin, also_bought): 
#    if(also_bought is None):
#        combined = [asin]
#    else:
#        combined = [asin] + also_bought
#    duplicates = ([item for item, count in Counter(combined).items() if count > 1])
#    return duplicates
#checkForDuplicatesUdf = udf(lambda asin, also_bought: 
#    checkForDuplicates(asin, also_bought), ArrayType(StringType()))
#boughtTranscationDf = metaRelatedDf_alsoBought.withColumn(
#        "items", checkForDuplicatesUdf("asin", "also_bought")).select("asin", "items")
#boughtTranscationDf.filter(size(col("items")) >= 1).show()

#===============================================================
## This method is to combine the items in a transaction together
#===============================================================
def combineItems(asin, also_bought):
    if(also_bought is None):
        combined = [asin]
    else:
        combined = [asin] + also_bought
    return list(set(combined)) #return only a unique set

combineItemsUdf = udf(lambda asin, also_bought: 
    combineItems(asin, also_bought), ArrayType(StringType()))
boughtTranscationDf = metaRelatedDf_alsoBought.withColumn(
        "items", combineItemsUdf("asin", "also_bought")).select("asin", "items")
#viewedTranscationDf = metaRelatedDf_alsoViewed.withColumn(
#        "items", combineItemsUdf("asin", "also_viewed")).select("asin", "items")

# split dataset
(trainingFP, testFP) = boughtTranscationDf.randomSplit([0.8, 0.2], seed=123)
testFP = testFP.withColumnRenamed("items", "also_bought")

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.005, minConfidence=0.5)
modelFPGrowth = fpGrowth.fit(trainingFP)

# Display frequent itemsets
freqItemsetsDf = modelFPGrowth.freqItemsets.sort("freq", ascending=False) # sorted by support
freqItemsetsDf_moreThan2 = freqItemsetsDf.filter(
        size(col("items"))>1).withColumnRenamed('items', 'frequent_item_set')
freqItemsetsDf_moreThan2.show(10)
#freqItemsetsDf_moreThan2.count() #220-(less than 2)->109

# Split frequent item set up cos u just want the first k based on freq
win_freq = Window.orderBy(desc("freq"))
freqItemsetsDf_moreThan2 = freqItemsetsDf_moreThan2.withColumn(
        "fisID", row_number().over(win_freq)).withColumn(
        "fis_item", explode(col("frequent_item_set")))

# Instead of using the transform function by FPGrowth, 
    # compare actual also_brought and frequent item set (contains itself)
win_asin = Window.partitionBy("asin").orderBy(desc("freq")) #ordered by highest support first
predictionsFP = freqItemsetsDf_moreThan2.join(
        testFP, freqItemsetsDf_moreThan2.fis_item==testFP.asin, "right").select(
                "asin", "also_bought", "fis_item", "fisID", "freq").withColumn(
                        "item_k", row_number().over(win_asin)).groupby("asin", "also_bought").agg(
                                collect_list("fis_item").alias('fis')).filter(
                                        size(col("fis"))>1)
#predictionsFP.show(10)
#predictionsFP.count()

k=2 # set k
print("k=" + str(k))

# Check if prediction column items are in items column
    # yes if at least a match, no if not, then calculate accuracy
def accuracyFP(also_bought, fis):
    if(len(fis) > k) :
        prediction = fis[1:k] #take the first k removing itself (sorted by freq)
    else :
        prediction = fis[1:len(fis)]
        
    if(len(set(prediction))< k) :
        return False
    else :
        intersectItems = (set(prediction)).intersection(set(also_bought))
        return len(intersectItems) >= 1
accuracyFPUdf = udf(lambda also_bought, fis: 
    accuracyFP(also_bought, fis), BooleanType())
predictionsCorrectnessFP = predictionsFP.withColumn(
        "matches", accuracyFPUdf("also_bought", "fis")).sort(
                "matches", ascending=True)
#predictionsCorrectnessFP.show(10)

# Calculate accuracy
totalCorrectRowsFP = predictionsCorrectnessFP.filter(
        predictionsCorrectnessFP["matches"]).count()
totalRowsFP = testFP.count()
accuracyFP = float(totalCorrectRowsFP)/float(totalRowsFP)
print("Accuracy of FP Growth = " + str(accuracyFP)) 

##################################
##
## 2. Collaborative filtering
##
##################################

reviewsDf = rawReviewDf.select("reviewerID", "asin", "overall")

# Show some summary
reviewsDf_userSummary = reviewsDf.groupby("reviewerID").count().distinct()
#reviewsDf_userSummary.show()
reviewsDf_allUsersSummary =  reviewsDf_userSummary.withColumnRenamed(
        "count", "reviewCount").groupby("reviewCount").count().sort(
                "reviewCount", ascending=True).withColumnRenamed(
                        "count", "users").limit(20)
reviewsDf_allUsersSummary.show()

# Plot graph to show user reviews count distribution for 1-20 reviews
layout_allUsersSummary=go.Layout(title="No of Users by Products Reviewed", 
                 xaxis={'title':'No of Products'}, 
                 yaxis={'title':'No of Users'})
bar_allUsersSummary = [go.Bar(
        x=reviewsDf_allUsersSummary.toPandas()['reviewCount'],
        y=reviewsDf_allUsersSummary.toPandas()['users'])]
fig_allUsersSummary = dict(data=bar_allUsersSummary, layout=layout_allUsersSummary)
plot_url_allUsersSummary = py.plot(
        fig_allUsersSummary, filename="reviewsDf_allUsersSummary",auto_open=False)
print(plot_url_allUsersSummary) #go to this url to see

# Filter only those with more than 2 reviews
reviewsDf = reviewsDf.join(reviewsDf_userSummary, ['reviewerID'], "right") \
    .filter("count > 2").drop('count')
reviewsDf.count()

## Assign numeric ID to reviewerID and asin
win_rev = Window.orderBy("reviewerID")
win_asin = Window.orderBy("asin")
reviewsDf = reviewsDf.withColumn("userID", dense_rank().over(win_rev)) \
    .withColumn("productID", dense_rank().over(win_asin))
reviewsDf.sort("reviewerID", ascending=False).show() #max userID = 768438
reviewsDf.sort("productID", ascending=False).show() #max productID = 166049

# split dataset
(trainingALS, testALS) = reviewsDf.randomSplit([0.8, 0.2], seed=123)

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="productID", ratingCol="overall",
          coldStartStrategy="drop")
modelALS = als.fit(trainingALS)

def accuracyALS(actual, recommendations):
    prediction = []
    if(not recommendations) :
        return False
    else :
        for item_score in recommendations:
            prediction.append(item_score[0]) # just want the item not rating
        intersectItems = (set(prediction)).intersection(set(actual))
        return len(intersectItems) >= 1
accuracyALSUdf = udf(lambda actual, recommendations: 
    accuracyALS(actual, recommendations), BooleanType())
    
als_result_x = [1,2,3,5,8,10]
als_result_y = []
#k=5 # set k
for k in als_result_x:
# Generate top k item recommendations for each user
    userRecs = modelALS.recommendForAllUsers(k)
    #userRecs.show() # 309930 for 6:4
    #userRecs.count()
    ## Generate top k user recommendations for each item - not needed
    #itemRecs = modelALS.recommendForAllItems(k)
    
    testALS_summary = (testALS.withColumn('actual', struct(testALS.productID)).groupBy('userID'). \
                       agg(collect_list('productID').alias('actual')))
    #testALS_summary.show()
    predictionsALS = userRecs.join(testALS_summary, testALS_summary.userID==userRecs.userID). \
                        select(testALS_summary.userID, "actual", "recommendations")
    #predictionsALS.show()
    
    print("k=" + str(k))
    # Check if prediction column items are in items column
        # yes if at least a match, no if not, then calculate accuracy
    
    predictionsCorrectnessALS = predictionsALS.withColumn(
            "matches", accuracyALSUdf("actual", "recommendations")).sort(
                    "matches", ascending=False)
    #predictionsCorrectnessALS.show()
    
    # Calculate accuracy
    totalCorrectRowsALS = predictionsCorrectnessALS.filter(
            predictionsCorrectnessALS["matches"]).count()
    totalRowsALS = testALS_summary.count()
    accuracyALS = float(totalCorrectRowsALS)/float(totalRowsALS)
    als_result_y.append((accuracyALS*100))
    print("Accuracy of ALS = " + str(accuracyALS)) 
    
# Plot graph to show Collective Filtering results
slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.array(als_result_x),np.array(als_result_y))
line = [(x * slope + intercept) for x in als_result_x]
layout_alsResults=go.Layout(
        title="Results of Collaborative Filtering using Alternate Least Squares", 
        xaxis={'title':'Value of K'}, 
        yaxis={'title':'Conversion Rate (%)', 'range':[0,0.05]})
line_alsResults = go.Scatter(
        x=als_result_x,
        y=als_result_y,
        mode='markers',
        marker=go.Marker(color='rgb(255, 127, 14)'),
        name='Data')
line_alsResultsFit = go.Scatter(
          x=als_result_x,
          y=line,
          mode='lines',
          marker=go.Marker(color='rgb(31, 119, 180)'),
          name='Fit'
    )
fig_alsResults = dict(data=[line_alsResults, line_alsResultsFit], 
                      layout=layout_alsResults)
plot_url_alsResults = py.plot(fig_alsResults, filename="resultsOfALS",auto_open=False)
print(plot_url_alsResults) # go to this url to see