# --------------------- SYSTEM & LIBRARIES ---------------------
import os
import glob
import pandas as pd
import findspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, concat_ws, lower, regexp_replace, size, split, length, when,
    avg, stddev, min, max
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# --------------------- SPARK ENVIRONMENT SETUP ---------------------
# Define paths to local Spark and Hadoop folders
spark_home = os.path.abspath(os.getcwd() + r"\OneDrive\Documents\spark\spark-3.5.5-bin-hadoop3")
hadoop_home = os.path.abspath(os.getcwd() + r"\OneDrive\Documents\spark\winutils")

# Set environment variables so Spark can find correct binaries
os.environ["SPARK_HOME"] = spark_home
os.environ["HADOOP_HOME"] = hadoop_home
os.environ["PATH"] = os.path.join(hadoop_home, "bin") + ";" + os.environ["PATH"]

# Initialize Spark context
findspark.init(spark_home)

# Create SparkSession = performance and memory settings
try:
    spark = SparkSession.builder \
        .appName("arxiv_socket_streaming") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "120s") \
        .config("spark.streaming.backpressure.enabled", "true") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("✅ SparkSession created successfully!")

except Exception as e:
    print(f"❌ Failed to create SparkSession: {e}")
    exit(1)

# --------------------- CLEAN & MERGE FILES ---------------------
# Read all .parquet files in the arxiv_data folder
input_folder = r"C:\Users\anast\OneDrive\Documents\arxiv_data"
parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))
cleaned_dfs = []

# Loop through and convert 'published' to datetime; skip bad files
for file in parquet_files:
    try:
        df = pd.read_parquet(file, engine="pyarrow")
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        cleaned_dfs.append(df)
    except Exception as e:
        print(f"⚠️ Skipped file {file} due to error: {e}")

# Merge all cleaned DataFrames into one big pandas DataFrame
final_df = pd.concat(cleaned_dfs, ignore_index=True)

# Save the merged, cleaned DataFrame to a new file
output_path = r"C:\Users\anast\OneDrive\Documents\arxiv_data_cleaned\arxiv_cleaned.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_parquet(output_path, engine="pyarrow", use_deprecated_int96_timestamps=True)
print(f"✅ Cleaned Parquet file saved: {output_path}")

# --------------------- LOAD DATA INTO SPARK ---------------------
# Load the cleaned parquet file into a Spark DataFrame
df = spark.read.parquet(output_path)

# --------------------- TEXT PREPROCESSING ---------------------
# Create a 'text' column by combining title and summary
df = df.withColumn("text", concat_ws(" ", col("title"), col("summary")))

# Lowercase the entire text
df = df.withColumn("text", lower(col("text")))

# Remove LaTeX expressions
df = df.withColumn("text", regexp_replace(col("text"), r"\\[a-zA-Z]+", ""))

# Remove non-alphabetic characters
df = df.withColumn("text", regexp_replace(col("text"), r"[^a-zA-Z\s]", " "))

# Replace multiple spaces with a single space
df = df.withColumn("text", regexp_replace(col("text"), r"\s+", " "))

# --------------------- FEATURE ENGINEERING ---------------------
# Count number of characters in the cleaned text
df = df.withColumn("char_length", length(col("text")))

# Count number of words in the title and summary separately
df = df.withColumn("title_length", size(split(col("title"), " ")))
df = df.withColumn("summary_length", size(split(col("summary"), " ")))

# Add binary feature: does the summary contain a URL
df = df.withColumn("has_url", when(col("summary").rlike("http[s]?://"), 1).otherwise(0))

# Add binary feature: does the summary mention code, python, or algorithm
df = df.withColumn("has_code", when(col("summary").rlike(r"\b(code|python|algorithm)\b"), 1).otherwise(0))

# --------------------- DATA EXPLORATION ---------------------
# Display total number of valid rows
print(f"✅ Number of rows: {df.count()}")

# Print schema to confirm column types
df.printSchema()

# Preview first 5 rows of the dataset
df.show(5, truncate=False)

# Show distribution of main categories (class imbalance)
print("📊 Category distribution:")
df.groupBy("main_category").count().orderBy("count", ascending=False).show(20, truncate=False)

# Statistics for character length of the text
print("📏 Character length stats:")
df.select("char_length").describe().show()

# Statistics for title and summary lenghts
print("📏 Title and summary length stats:")
df.select(
    avg("title_length").alias("avg_title_len"),
    stddev("title_length").alias("std_title_len"),
    min("title_length").alias("min_title_len"),
    max("title_length").alias("max_title_len"),
    avg("summary_length").alias("avg_summary_len"),
    stddev("summary_length").alias("std_summary_len"),
    min("summary_length").alias("min_summary_len"),
    max("summary_length").alias("max_summary_len")
).show()

# --------------------- CORRELATION MATRIX ---------------------
# Create a vector of selected numeric features
numeric_cols = ["char_length", "title_length", "summary_length", "has_url", "has_code"]
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
df_vector = assembler.transform(df)

# Compute pearson correlation matrix
corr_matrix = Correlation.corr(df_vector, "features_vec", method="pearson").head()[0]
print("📈 Correlation matrix:")
print(corr_matrix)

# --------------------- OUTLIERS DETECTION ANS CLEANING ---------------------
from pyspark.sql.functions import col, split, size, slice, array_join, percentile_approx

# Remove extremely short titles (1 word or less)
df = df.filter(col("title_length") > 1)

# Truncate long titles to first 20 words (upper IQR bound is 20.5, so 20 is safe)
df = df.withColumn("title", array_join(slice(split(col("title"), " "), 1, 20), " "))

# Recalculate title_length based on the truncated version
df = df.withColumn("title_length", size(split(col("title"), " ")))

# --------------------- OUTLIER CHECK AFTER CLEANING ---------------------
# List of numeric columns to check
numeric_cols = ["char_length", "title_length", "summary_length"]

for col_name in numeric_cols:
    # Compute Q1 and Q3
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
    q1, q3 = quantiles
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f"\n📊 Outlier thresholds for '{col_name}':")
    print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

    # Count how many outliers exist
    count_outliers = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()
    print(f"🚨 Number of outliers in '{col_name}': {count_outliers}")

# Group category counts and convert to pandas
category_counts = df.groupBy("main_category").count().orderBy("count", ascending=False).toPandas()

# Plot bar chart
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(category_counts["main_category"], category_counts["count"], color="cornflowerblue")
plt.xticks(rotation=90)
plt.xlabel("Main Category")
plt.ylabel("Number of Articles")
plt.title("Distribution of Articles by Main Category")
plt.tight_layout()
plt.show()

# -------------------- COMPUTE CLASS WEIGHTS --------------------
from pyspark.sql.functions import col, lit

label_counts = df.groupBy("main_category").count()
total_count = df.count()

label_weights = label_counts.withColumn(
    "class_weight",
    lit(total_count) / col("count")
)

df_weighted = df.join(label_weights.select("main_category", "class_weight"), on="main_category", how="left")

# -------------------- SPLIT DATA --------------------
train, test = df_weighted.randomSplit([0.8, 0.2], seed=42)

# -------------------- CLASSIFIER WITH WEIGHT --------------------
from pyspark.ml.classification import LogisticRegression

classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="class_weight", 
    maxIter=100,
    regParam=0.01
)

# -------------------- BUILD PIPELINE WITH THIS CLASSIFIER --------------------
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer,
    VectorAssembler, StandardScaler
)

tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="text_features")

numeric_features = ["char_length", "title_length", "summary_length"]
numeric_assembler = VectorAssembler(inputCols=numeric_features, outputCol="numeric_vec")
scaler = StandardScaler(inputCol="numeric_vec", outputCol="scaled_numeric", withMean=True, withStd=True)

binary_features = ["has_url", "has_code"]
assembler = VectorAssembler(
    inputCols=["text_features", "scaled_numeric"] + binary_features,
    outputCol="features"
)

label_indexer = StringIndexer(inputCol="main_category", outputCol="label", handleInvalid="keep")

pipeline = Pipeline(stages=[
    tokenizer,
    stopwords,
    hashingTF,
    idf,
    numeric_assembler,
    scaler,
    assembler,
    label_indexer,
    classifier 
])

# -------------------- CrossValidator and evaluation --------------------
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

paramGrid = ParamGridBuilder() \
    .addGrid(classifier.maxIter, [50, 100]) \
    .addGrid(classifier.regParam, [0.01, 0.1]) \
    .build()

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2
)

# -------------------- TRAIN AND TEST --------------------
cv_model = cv.fit(train)
predictions = cv_model.transform(test)

f1 = evaluator.evaluate(predictions)
print(f"✅ Logistic Regression F1 Score: {f1:.4f}")

best_model = cv_model.bestModel.stages[-1]
print("📌 Best maxIter:", best_model._java_obj.getMaxIter())
print("📌 Best regParam:", best_model._java_obj.getRegParam())

#-------------------------------------weights-----------------------
# Check if weighted metrics are hiding  per-class performance, i think it is suspicious we end up with 0.89 accuracy, sounds to good to be true
# F1 score (already computed)
evaluator.setMetricName("f1")
f1 = evaluator.evaluate(predictions)
print(f"⚖️ Weighted F1 Score: {f1:.4f}")

# Accuracy
evaluator.setMetricName("accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"📈 Accuracy: {accuracy:.4f}")

# Weighted Precision
evaluator.setMetricName("weightedPrecision")
weighted_precision = evaluator.evaluate(predictions)
print(f"🎯 Weighted Precision: {weighted_precision:.4f}")

# Weighted Recall
evaluator.setMetricName("weightedRecall")
weighted_recall = evaluator.evaluate(predictions)
print(f"🔁 Weighted Recall: {weighted_recall:.4f}")

# View the confusion matrix to assess per-class predictions
# Confusion matrix counts
print("📊 Confusion matrix:")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show(100)

# Recover label to category mapping from the StringIndexer
fitted_pipeline = cv_model.bestModel
string_indexer = fitted_pipeline.stages[-2]
category_labels = string_indexer.labels

# Display mappings
print("🔖 Label index mapping:")
for i, label in enumerate(category_labels):
    print(f"Index {i} -> {label}")

#///////////////////////////////////////////////////////////////////////////////////////////////////////
# SAVE THE MODEL
from pyspark.ml import PipelineModel

# Save trained model (NOT in OneDrive this time)
cv_model.bestModel.write().overwrite().save("file:///C:/anastasia_bigdata/arxiv_model")
print("✅ Trained model saved to disk in C:/anastasia_bigdata/arxiv_model")

# Save the final cleaned + preprocessed dataset used for training/testing (NOT in OneDrive my mistake)
df_weighted.write.mode("overwrite").parquet("file:///C:/anastasia_bigdata/final_preprocessed_data.parquet")
print("✅ Final preprocessed dataset saved to disk in C:/anastasia_bigdata/final_preprocessed_data.parquet")

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#LIVESTREAM OMG
from pyspark.sql.functions import from_json, col, concat_ws, lower, regexp_replace, length, size, split, when
from pyspark.sql.types import StructType, StringType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString

# Load trained model from safe local folder
model = PipelineModel.load("file:///C:/anastasia_bigdata/arxiv_model")
print("✅ Model loaded. Ready to stream.")

# Define schema for the incoming JSON stream
schema = StructType() \
    .add("aid", StringType()) \
    .add("title", StringType()) \
    .add("summary", StringType()) \
    .add("main_category", StringType()) \
    .add("categories", StringType()) \
    .add("published", StringType())

# Connect to the server
stream_df = spark.readStream \
    .format("socket") \
    .option("host", "seppe.net") \
    .option("port", 7778) \
    .load()

# Parse the incoming JSON strings
parsed_df = stream_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Repeat the same preprocessing as in training
preprocessed_df = parsed_df \
    .withColumn("text", concat_ws(" ", col("title"), col("summary"))) \
    .withColumn("text", lower(col("text"))) \
    .withColumn("text", regexp_replace(col("text"), r"\\[a-zA-Z]+", "")) \
    .withColumn("text", regexp_replace(col("text"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("text", regexp_replace(col("text"), r"\s+", " ")) \
    .withColumn("char_length", length(col("text"))) \
    .withColumn("title_length", size(split(col("title"), " "))) \
    .withColumn("summary_length", size(split(col("summary"), " "))) \
    .withColumn("has_url", when(col("summary").rlike("http[s]?://"), 1).otherwise(0)) \
    .withColumn("has_code", when(col("summary").rlike(r"\b(code|python|algorithm)\b"), 1).otherwise(0))

# Make predictions with model
predictions = model.transform(preprocessed_df)

# Convert prediction index back to category label
label_decoder = IndexToString(
    inputCol="prediction",
    outputCol="predicted_category",
    labels=model.stages[-2].labels
)
decoded_predictions = label_decoder.transform(predictions)

# Output predictions to console
query = decoded_predictions.select("aid", "title", "predicted_category").writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
