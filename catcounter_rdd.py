from pyspark import SparkContext, SparkConf
import json

# Create a SparkContext
conf = SparkConf().setAppName("CategoriesCounter")
sc = SparkContext(conf=conf)

# Load the input file from HDFS
input_file = sc.textFile("hdfs:///user/dic23_shared/amazon-reviews/full/reviews_devset.json")

# Here we read the list of stopwords that must be ignored from the review text stored as global variable
stop_words = open('stopwords.txt').readlines()
stop_words = [w.replace('\n', '') for w in open('stopwords.txt').readlines()]

# Parse each line as JSON
data = input_file.map(json.loads)

# Map operation: Extract category as key and emit count of 1
category_counts = data.map(lambda x: (x["category"], 1))

# Reduce operation: Sum the counts for each category
category_counts = category_counts.reduceByKey(lambda x, y: x + y)

# Filter out stopwords
category_counts = category_counts.filter(lambda x: x[0] not in stop_words)

# Collect the results
results = category_counts.collect()

# Print the results
for result in results:
    print(f'"{result[0]}"\t{result[1]}')

# Stop Spark
sc.stop()
