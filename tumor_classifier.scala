/*

Predicting Tumor Malignancy with Spark ML

Dataset: 
"cancer.csv"

Algorithm:
Random Forest

*/


--upload 'cancer.csv'

--view dataset
head cancer.csv


--make directory and copy the dataset to HDFS

hadoop fs -mkdir /BigData/
hadoop fs -copyFromLocal cancer.csv /BigData/.

hadoop fs -ls /BigData/


-- Run spark-shell

spark-shell

/*
--import statements
*/

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{IntegerType, DoubleType}

/*
--read csv file
*/

val raw_dataset = spark.read
 .format("csv")
 .option("header", "true")
 .load("hdfs://10.128.0.7:8020/BigData/cancer.csv")


raw_dataset.show()

raw_dataset.printSchema()


/*
-- Select needed columns and cast the date type from string to integer
*/

val dataset = raw_dataset.select(
 col("Clump Thickness").cast("int"), 
 col("UofCSize").cast("int"), 
 col("UofCShape").cast("int"), 
 col("Marginal Adhesion").cast("int"), 
 col("SECSize").cast("int"), 
 col("Bare Nuclei").cast("int"), 
 col("Bland Chromatin").cast("int"), 
 col("Normal Nucleoli").cast("int"), 
 col("Mitoses").cast("int"),
 col("Class").cast("int"))


dataset.printSchema()

/*
--check null Values
*/

dataset.select(dataset.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

/*
--check Min & Max
*/

dataset.describe().show()


/*
-- Split dataset into training and test data typical 80 20
-- give a seed: 754
*/

val Array(trainingData, testData) = dataset.randomSplit(Array(0.8, 0.2), 754) 


/*
--Assemble all features
*/

val assembler = new VectorAssembler()
 .setInputCols(Array("Clump Thickness", "UofCSize", "UofCShape", "Marginal Adhesion", "SECSize", "Bare Nuclei", "Bland Chromatin","Normal Nucleoli","Mitoses" ))
 .setOutputCol("assembled-features")

/*Random Forest  
-- create a new random forest object 
-- give features as "assembled-features"
-- give the label as "Class"
*/

val rf = new RandomForestClassifier()
 .setFeaturesCol("assembled-features")
 .setLabelCol("Class")
 .setSeed(1234)
  
/*Set up pipeline
-- Use pipepline to set our stages
-- So our stages are the vector assembler and the random forest classifier object
*/

val pipeline = new Pipeline()
  .setStages(Array(assembler, rf))

/*To evaluate the model
-- Here we are using MulticlassClassificationEvaluator
-- we compared "Class" with the prediction column
-- metric "accuracy" is basically percentage of accurate results
*/

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("Class")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

/*
-- Hyper parameters regarding maxDepth and impurity
*/

val paramGrid = new ParamGridBuilder()  
  .addGrid(rf.maxDepth, Array(2,3,4,5))
  .addGrid(rf.impurity, Array("entropy","gini")).build()

/*Cross validate model
-- tie everything together with cross-validator
-- set the pipeline for estimator
-- Multiclassevaluator for evaluator
-- Set the hyper parameters and the number of folds
*/

val cross_validator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

/*
-- Train the model on training data
-- Gives us the best model from the 6 variations
*/

val cvModel = cross_validator.fit(trainingData)


/*Predict with test data
-- Transform testData with predictions
*/

val predictions = cvModel.transform(testData)


/*Evaluate the model
-- check with actual values and print accuracy
*/

val accuracy = evaluator.evaluate(predictions)

println("accuracy on test data = " + accuracy)
