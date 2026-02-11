# Customer Churn Prediction using Spark ML on Amazon EMRProject
Here is the project documentation for your README, formatted with the exact header styles you requested and including code comments for better explanation.

Customer Churn Prediction using Spark ML on Amazon EMR
Project Description
This project implements a distributed machine learning pipeline to predict bank customer churn. By using Amazon EMR, we leverage the power of Apache Spark to process data in a distributed cluster environment, ensuring the model can scale with increasing data volumes.

Dataset Information
Source: Bank Customer Churn Dataset (Kaggle)

URL: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

Target Variable: Exited (1 = Churn, 0 = Stayed)

Features: CreditScore, Geography, Gender, Age, Balance, Tenure, etc.

ML Pipeline Stages
The solution is built using the mandatory Spark ML Pipeline abstraction:

StringIndexer: Encodes Geography and Gender into numerical indices.

OneHotEncoder: Maps indices to binary vectors.

VectorAssembler: Merges all feature columns into a single "features" vector.

StandardScaler: Normalizes data for zero mean and unit variance.

Estimator: Logistic Regression (Baseline) or Random Forest (Experiment).

Setup Instructions
1. HDFS Configuration
Move the data from the master node local storage into the distributed file system:

Bash
# Create the input directory in HDFS
hdfs dfs -mkdir -p /user/hadoop/churn_input

# Move the local CSV file to the HDFS directory
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/
2. Execution Commands
Run the following commands on the EMR master node:

Run Baseline (Logistic Regression):

Bash
# Submitting the baseline job to YARN in client mode
spark-submit --master yarn --deploy-mode client churn_pipeline.py
Run Experiment (Random Forest):

Bash
# Submitting the comparison experiment to YARN
spark-submit --master yarn --deploy-mode client churn_pipeline_rf.py
Experiment Results
I conducted an experiment comparing Logistic Regression and Random Forest. Based on the execution logs for the Random Forest model:

Accuracy: 0.8412

Precision: 0.8359

Recall: 0.8412

F1 Score: 0.8109

Observations
Scalability: The Random Forest model performed effectively in the distributed environment.

Parallelism: Running this on EMR allowed the workload to be partitioned across multiple worker nodes, which was monitored and verified via the YARN UI.

Accuracy: The ensemble approach of Random Forest typically handles the non-linear features of the bank dataset better than simple linear regression.
