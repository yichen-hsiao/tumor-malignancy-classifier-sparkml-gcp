# Predicting Tumor Malignancy with Spark ML

## Objective

This portfolio project demonstrates the development of a scalable machine learning pipeline for predicting tumor malignancy using a cancer dataset in a big data ecosystem powered by Apache Spark on Google Cloud Platform (GCP).

## Method

### Tools: 
Google Cloud Platform (GCP), HDFS, Apache Spark MLlib

### Language:
Scala

### Machine Learning Algorithm:
Random Forest Classifier

### Dataset:

**Source:** ​[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

**Original Size:** 699 rows

**Usable Size (after cleaning):** 683 observations

**Variables:**

- Sample code number (id number)
- Clump Thickness (1-10)
- Uniformity of Cell Size (1- 10)
- Uniformity of Cell Shape (1-10)
- Marginal Adhesion (1-10)
- Single Epithelial Cell Size (1-10)
- Bare Nuclei (1-10)
- Bland Chromatin (1-10)
- Normal Nucleoli (1-10)
- Mitoses (1-10)
- Class (benign/malignant)​

## Development Steps
1. Load data into HDFS and read into Spark Shell

2. Inspect schema and summarize dataset in Spark

3. Split data into training and test sets (80/20)

4. Assemble features into a single vector

5. Create a Random Forest classifier with input/output columns

6. Build a pipeline combining feature assembly and model fitting

7. Evaluate model and tune hyperparameters

8. Perform cross-validation

9. Predict on unseen test data

10. Calculate model performance metrics

## Main Finding
The Random Forest model achieved an accuracy of 0.97 on unseen data, highlighting its strong predictive capability. Although the training was conducted on a relatively small dataset, the use of a big data infrastructure ensures that the model and pipeline are inherently scalable. This framework can be readily adapted for similar datasets and extended to future experiments involving larger or more complex data.

