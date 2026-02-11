from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Инициализация Spark
spark = SparkSession.builder.appName("CustomerChurnPipeline").getOrCreate()

print("=" * 50)
print("Loading data from HDFS...")
print("=" * 50)

# Загрузка данных
data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

print(f"Total records: {data.count()}")
print("Schema:")
data.printSchema()

# Показать примеры данных
print("\nSample data:")
data.show(5)

# Разделение на train/test
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print(f"\nTraining set: {train_data.count()} records")
print(f"Test set: {test_data.count()} records")

# Stage 1: StringIndexer для категориальных признаков
geo_indexer = StringIndexer(
    inputCol="Geography",
    outputCol="GeographyIndex",
    handleInvalid="keep"
)

gender_indexer = StringIndexer(
    inputCol="Gender",
    outputCol="GenderIndex",
    handleInvalid="keep"
)

# Stage 2: OneHotEncoder
encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

# Stage 3: VectorAssembler
assembler = VectorAssembler(
    inputCols=[
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary",
        "GeographyVec", "GenderVec"
    ],
    outputCol="features",
    handleInvalid="skip"
)

# Stage 4: StandardScaler
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=False,
    withStd=True
)

# Stage 5: Logistic Regression
rf = RandomForestClassifier(
    labelCol="Exited",
    featuresCol="scaledFeatures",
    numTrees=20,
    maxDepth=5
)

# Создание Pipeline
print("\n" + "=" * 50)
print("Building ML Pipeline...")
print("=" * 50)

pipeline = Pipeline(stages=[
    geo_indexer,
    gender_indexer,
    encoder,
    assembler,
    scaler,
    rf
])

# Обучение модели
print("\n" + "=" * 50)
print("Training Random Forest model...")
print("=" * 50)
import time
start_time = time.time()

model = pipeline.fit(train_data)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Предсказания на тестовой выборке
print("\nMaking predictions on test set...")
predictions = model.transform(test_data)

# Показать примеры предсказаний
print("\nSample predictions:")
predictions.select("Exited", "prediction", "probability").show(10, truncate=False)

# Оценка модели
print("\n" + "=" * 50)
print("Model Evaluation")
print("=" * 50)

# Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = accuracy_evaluator.evaluate(predictions)

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision = precision_evaluator.evaluate(predictions)

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall = recall_evaluator.evaluate(predictions)

# F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
predictions.groupBy("Exited", "prediction").count().show()

print("\n" + "=" * 50)
print("Pipeline execution completed successfully!")
print("=" * 50)

spark.stop()