# %% [markdown]
# # NYC Taxi Trip Analytics using Apache Spark
# ## Big Data Analytics (DS-313) — Assignment 3
# **Student Name:** [YOUR NAME]
# **Roll Number:** [YOUR ROLL NUMBER]
# **Date:** August 2026

# %%
# ============================================================
# PART 1: ENVIRONMENT SETUP (10 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/spark_config.png
#    → Take screenshot of the FULL output of this cell
# ============================================================

import sys
import os
import platform
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import time

# Create Spark Session
spark = SparkSession.builder \
    .appName("NYC Taxi Trip Analytics") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.ui.port", "4040") \
    .getOrCreate()

print("=" * 60)
print("        ENVIRONMENT SETUP — SYSTEM INFORMATION")
print("=" * 60)
print(f"  Python Version    : {sys.version.split()[0]}")
print(f"  Spark Version     : {spark.version}")
print(f"  Operating System  : {platform.system()} {platform.release()}")
print(f"  Architecture      : {platform.machine()}")
print(f"  Processor         : {platform.processor()}")
print(f"  Java Home         : {os.environ.get('JAVA_HOME', 'Not set')}")
print(f"  Spark Master      : {spark.sparkContext.master}")
print(f"  App Name          : {spark.sparkContext.appName}")
print(f"  Default Parallelism: {spark.sparkContext.defaultParallelism}")
print("=" * 60)
print("\n📋 Spark Configuration:")
for key, value in sorted(spark.sparkContext.getConf().getAll()):
    print(f"    {key} = {value}")
print(f"\n🌐 Spark UI available at: http://localhost:4040")

# %% [markdown]
# ---
# ## Part 2: Load Dataset (10 Marks)

# %%
# ============================================================
# PART 2: LOAD DATASET (10 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/schema.png
#    → Take screenshot of the schema + record count output
# ============================================================

# Load the Parquet file — UPDATE THE PATH if your file is named differently
df = spark.read.parquet("yellow_tripdata_2024-01.parquet")

print("=" * 60)
print("        DATASET LOADED SUCCESSFULLY")
print("=" * 60)

# Display Schema
print("\n📋 SCHEMA:")
df.printSchema()

# Dataset dimensions
record_count = df.count()
print(f"\n📊 Number of Columns : {len(df.columns)}")
print(f"📊 Number of Records : {record_count:,}")
print(f"\n📋 Column Names: {df.columns}")

# %%
# Display first 20 rows
# 📸 SCREENSHOT: screenshots/first_20_rows.png
print("📋 FIRST 20 ROWS:")
df.show(20, truncate=False)

# %% [markdown]
# ---
# ## Part 3: Exploratory Data Analysis (15 Marks)

# %%
# ============================================================
# PART 3: EXPLORATORY DATA ANALYSIS (15 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/eda_results.png
#    → Take screenshot of ALL 10 answers below
# ============================================================

print("=" * 60)
print("        EXPLORATORY DATA ANALYSIS")
print("=" * 60)

total_trips = df.count()
earliest_date = df.select(min("tpep_pickup_datetime")).collect()[0][0]
latest_date = df.select(max("tpep_pickup_datetime")).collect()[0][0]
unique_vendors = df.select("VendorID").distinct().count()
avg_distance = df.select(avg("trip_distance")).collect()[0][0]
avg_fare = df.select(avg("fare_amount")).collect()[0][0]
max_fare = df.select(max("fare_amount")).collect()[0][0]
min_fare = df.select(min("fare_amount")).collect()[0][0]
avg_passengers = df.select(avg("passenger_count")).collect()[0][0]
payment_methods = df.select("payment_type").distinct().count()

print(f"\n  1.  Total Trips             : {total_trips:,}")
print(f"  2.  Earliest Trip Date      : {earliest_date}")
print(f"  3.  Latest Trip Date        : {latest_date}")
print(f"  4.  Unique Vendors          : {unique_vendors}")
print(f"  5.  Average Trip Distance   : {avg_distance:.2f} miles")
print(f"  6.  Average Fare            : ${avg_fare:.2f}")
print(f"  7.  Maximum Fare            : ${max_fare:.2f}")
print(f"  8.  Minimum Fare            : ${min_fare:.2f}")
print(f"  9.  Average Passenger Count : {avg_passengers:.2f}")
print(f"  10. Number of Payment Methods: {payment_methods}")
print("\n" + "=" * 60)

# %% [markdown]
# ---
# ## Part 4: Data Cleaning (10 Marks)

# %%
# ============================================================
# PART 4: DATA CLEANING (10 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/data_cleaning.png
#    → Take screenshot of before/after counts
# ============================================================

print("=" * 60)
print("        DATA CLEANING")
print("=" * 60)

original_count = df.count()
print(f"\n🔴 Before Cleaning: {original_count:,} records")

# Step 1: Remove Duplicates
# Duplicates can skew analysis results and inflate trip counts
df_clean = df.dropDuplicates()
after_dedup = df_clean.count()
print(f"\n✅ Step 1 — Remove Duplicates")
print(f"   Removed: {original_count - after_dedup:,} duplicate rows")
print(f"   Remaining: {after_dedup:,} records")

# Step 2: Remove Invalid Trip Distances (zero or negative)
# Trips with 0 or negative distance are data errors
before = df_clean.count()
df_clean = df_clean.filter(col("trip_distance") > 0)
after = df_clean.count()
print(f"\n✅ Step 2 — Remove Invalid Trip Distances (distance <= 0)")
print(f"   Removed: {before - after:,} invalid rows")
print(f"   Remaining: {after:,} records")

# Step 3: Remove Negative Fares
# Negative fares are billing errors or refunds
before = df_clean.count()
df_clean = df_clean.filter(col("fare_amount") >= 0)
after = df_clean.count()
print(f"\n✅ Step 3 — Remove Negative Fares")
print(f"   Removed: {before - after:,} invalid rows")
print(f"   Remaining: {after:,} records")

# Step 4: Handle Missing Values
# Drop rows where critical columns have null values
before = df_clean.count()
critical_cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
                 "passenger_count", "trip_distance", "fare_amount", "total_amount"]
df_clean = df_clean.na.drop(subset=critical_cols)
after = df_clean.count()
print(f"\n✅ Step 4 — Handle Missing Values (drop nulls in critical columns)")
print(f"   Removed: {before - after:,} rows with missing values")
print(f"   Remaining: {after:,} records")

print(f"\n{'=' * 60}")
print(f"🟢 After Cleaning: {df_clean.count():,} records")
print(f"   Total Removed : {original_count - df_clean.count():,} records")
print(f"   Retention Rate: {df_clean.count()/original_count*100:.1f}%")
print(f"{'=' * 60}")

# %% [markdown]
# ---
# ## Part 5: Spark Transformations (15 Marks)
# Using at least 10 transformations: filter, select, withColumn, orderBy, drop,
# distinct, groupBy, join, alias, repartition

# %%
# ============================================================
# PART 5: SPARK TRANSFORMATIONS (15 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/transformations.png
#    → Take screenshot(s) of each transformation output
#    → You may need multiple screenshots, name them:
#      screenshots/transform_1_filter.png
#      screenshots/transform_2_select.png
#      ... etc.
# ============================================================

print("=" * 60)
print("        SPARK TRANSFORMATIONS")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Transformation 1: filter()
# Purpose: Filter trips with distance greater than 5 miles
#          to analyze long-distance taxi rides
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 1: filter()")
print("   Purpose: Select trips longer than 5 miles")
df_filtered = df_clean.filter(col("trip_distance") > 5)
print(f"   Long trips (>5 miles): {df_filtered.count():,}")
df_filtered.select("VendorID", "trip_distance", "fare_amount", "total_amount").show(5)

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 2: select()
# Purpose: Select only relevant columns for focused analysis
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 2: select()")
print("   Purpose: Select key columns for analysis")
df_selected = df_clean.select("VendorID", "tpep_pickup_datetime",
                               "trip_distance", "fare_amount",
                               "tip_amount", "total_amount", "payment_type")
df_selected.show(5)

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 3: withColumn()
# Purpose: Create new calculated columns — trip duration in
#          minutes and cost per mile
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 3: withColumn()")
print("   Purpose: Add trip_duration_min and cost_per_mile columns")
df_enriched = df_clean \
    .withColumn("trip_duration_min",
                round((unix_timestamp("tpep_dropoff_datetime") -
                       unix_timestamp("tpep_pickup_datetime")) / 60, 2)) \
    .withColumn("cost_per_mile",
                round(col("fare_amount") / col("trip_distance"), 2))
df_enriched.select("trip_distance", "fare_amount", "trip_duration_min", "cost_per_mile").show(5)

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 4: orderBy()
# Purpose: Find the most expensive trips by sorting
#          in descending order of total_amount
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 4: orderBy()")
print("   Purpose: Sort by total amount (descending) — top 10 most expensive")
df_sorted = df_clean.orderBy(col("total_amount").desc())
df_sorted.select("VendorID", "trip_distance", "fare_amount", "tip_amount", "total_amount").show(10)

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 5: drop()
# Purpose: Remove unnecessary columns that are not needed
#          for our analysis
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 5: drop()")
print("   Purpose: Remove store_and_fwd_flag and Airport_fee columns")
df_dropped = df_clean.drop("store_and_fwd_flag", "Airport_fee")
print(f"   Columns BEFORE drop: {len(df_clean.columns)} → {df_clean.columns}")
print(f"   Columns AFTER  drop: {len(df_dropped.columns)} → {df_dropped.columns}")

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 6: distinct()
# Purpose: Find unique vendor IDs and payment types
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 6: distinct()")
print("   Purpose: Find unique Vendor IDs and Payment Types")
print("\n   Unique Vendor IDs:")
df_clean.select("VendorID").distinct().show()
print("   Unique Payment Types:")
df_clean.select("payment_type").distinct().orderBy("payment_type").show()

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 7: groupBy()
# Purpose: Aggregate trips and revenue by payment type
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 7: groupBy()")
print("   Purpose: Trip count and average fare grouped by payment type")
df_grouped = df_clean.groupBy("payment_type").agg(
    count("*").alias("trip_count"),
    round(avg("fare_amount"), 2).alias("avg_fare"),
    round(sum("total_amount"), 2).alias("total_revenue"),
    round(avg("tip_amount"), 2).alias("avg_tip")
)
df_grouped.orderBy("payment_type").show()

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 8: join()
# Purpose: Join trip data with a vendor lookup table
#          to get human-readable vendor names
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 8: join()")
print("   Purpose: Join with vendor names lookup table")

# Create a vendor lookup DataFrame
vendor_data = [(1, "Creative Mobile Technologies"),
               (2, "VeriFone Inc.")]
vendor_schema = StructType([
    StructField("VendorID", LongType(), True),
    StructField("vendor_name", StringType(), True)
])
vendors_df = spark.createDataFrame(vendor_data, schema=vendor_schema)

df_joined = df_clean.join(vendors_df, "VendorID", "left")
df_joined.select("VendorID", "vendor_name", "trip_distance", "fare_amount").show(5)

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 9: alias()
# Purpose: Rename columns for better readability
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 9: alias()")
print("   Purpose: Rename columns for clarity")
df_aliased = df_clean.select(
    col("VendorID").alias("Vendor_ID"),
    col("tpep_pickup_datetime").alias("Pickup_Time"),
    col("trip_distance").alias("Distance_Miles"),
    col("fare_amount").alias("Fare_USD"),
    col("tip_amount").alias("Tip_USD"),
    col("total_amount").alias("Total_USD")
)
df_aliased.show(5)

# %%
# ──────────────────────────────────────────────────────────────
# Transformation 10: repartition()
# Purpose: Repartition data for parallel processing
# ──────────────────────────────────────────────────────────────
print("\n📌 Transformation 10: repartition()")
print("   Purpose: Change number of partitions for parallelism")
print(f"   Partitions BEFORE: {df_clean.rdd.getNumPartitions()}")
df_repartitioned = df_clean.repartition(8)
print(f"   Partitions AFTER : {df_repartitioned.rdd.getNumPartitions()}")

# %% [markdown]
# ---
# ## Part 6: Spark SQL (15 Marks)
# Create a temporary view and write at least 10 SQL queries

# %%
# ============================================================
# PART 6: SPARK SQL (15 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/sql_queries.png
#    → Take screenshot(s) of each query result
#    → You may need multiple screenshots, name them:
#      screenshots/sql_q1.png, screenshots/sql_q2.png, etc.
# ============================================================

# Create a temporary SQL view
df_clean.createOrReplaceTempView("taxi_trips")

print("=" * 60)
print("        SPARK SQL QUERIES")
print("=" * 60)
print("✅ Temporary view 'taxi_trips' created successfully\n")

# %%
# ──── SQL Query 1: Top 10 Longest Trips ────
print("📌 SQL Query 1: Top 10 Longest Trips")
q1 = spark.sql("""
    SELECT trip_distance,
           fare_amount,
           total_amount,
           tpep_pickup_datetime,
           tpep_dropoff_datetime
    FROM taxi_trips
    ORDER BY trip_distance DESC
    LIMIT 10
""")
q1.show(truncate=False)

# %%
# ──── SQL Query 2: Top 10 Pickup Locations ────
print("📌 SQL Query 2: Top 10 Pickup Locations by Trip Count")
q2 = spark.sql("""
    SELECT PULocationID AS pickup_location,
           COUNT(*) AS trip_count,
           ROUND(AVG(fare_amount), 2) AS avg_fare,
           ROUND(AVG(trip_distance), 2) AS avg_distance
    FROM taxi_trips
    GROUP BY PULocationID
    ORDER BY trip_count DESC
    LIMIT 10
""")
q2.show()

# %%
# ──── SQL Query 3: Average Fare by Payment Type ────
print("📌 SQL Query 3: Average Fare by Payment Type")
q3 = spark.sql("""
    SELECT payment_type,
           CASE payment_type
               WHEN 1 THEN 'Credit Card'
               WHEN 2 THEN 'Cash'
               WHEN 3 THEN 'No Charge'
               WHEN 4 THEN 'Dispute'
               WHEN 5 THEN 'Unknown'
               ELSE 'Other'
           END AS payment_name,
           COUNT(*) AS total_trips,
           ROUND(AVG(fare_amount), 2) AS avg_fare,
           ROUND(AVG(tip_amount), 2) AS avg_tip,
           ROUND(SUM(total_amount), 2) AS total_revenue
    FROM taxi_trips
    GROUP BY payment_type
    ORDER BY total_trips DESC
""")
q3.show(truncate=False)

# %%
# ──── SQL Query 4: Peak Pickup Hour ────
print("📌 SQL Query 4: Peak Pickup Hours")
q4 = spark.sql("""
    SELECT HOUR(tpep_pickup_datetime) AS pickup_hour,
           COUNT(*) AS trip_count,
           ROUND(AVG(fare_amount), 2) AS avg_fare
    FROM taxi_trips
    GROUP BY HOUR(tpep_pickup_datetime)
    ORDER BY trip_count DESC
""")
q4.show(24)

# %%
# ──── SQL Query 5: Trips Over 20 Miles ────
print("📌 SQL Query 5: Trips Over 20 Miles")
q5 = spark.sql("""
    SELECT COUNT(*) AS long_trips,
           ROUND(AVG(trip_distance), 2) AS avg_distance,
           ROUND(AVG(fare_amount), 2) AS avg_fare,
           ROUND(MIN(fare_amount), 2) AS min_fare,
           ROUND(MAX(fare_amount), 2) AS max_fare,
           ROUND(SUM(total_amount), 2) AS total_revenue
    FROM taxi_trips
    WHERE trip_distance > 20
""")
q5.show()

# %%
# ──── SQL Query 6: Daily Revenue (Monthly Revenue) ────
print("📌 SQL Query 6: Daily Revenue Summary")
q6 = spark.sql("""
    SELECT DATE(tpep_pickup_datetime) AS trip_date,
           COUNT(*) AS total_trips,
           ROUND(SUM(total_amount), 2) AS daily_revenue,
           ROUND(AVG(total_amount), 2) AS avg_trip_revenue
    FROM taxi_trips
    GROUP BY DATE(tpep_pickup_datetime)
    ORDER BY trip_date
""")
q6.show(31)

# %%
# ──── SQL Query 7: Average Tip Percentage by Payment Type ────
print("📌 SQL Query 7: Average Tip Percentage by Payment Type")
q7 = spark.sql("""
    SELECT payment_type,
           ROUND(AVG(tip_amount), 2) AS avg_tip,
           ROUND(AVG(CASE WHEN fare_amount > 0
                     THEN (tip_amount / fare_amount) * 100
                     ELSE 0 END), 2) AS avg_tip_pct
    FROM taxi_trips
    GROUP BY payment_type
    ORDER BY avg_tip_pct DESC
""")
q7.show()

# %%
# ──── SQL Query 8: Trips by Passenger Count ────
print("📌 SQL Query 8: Trips by Passenger Count")
q8 = spark.sql("""
    SELECT CAST(passenger_count AS INT) AS passengers,
           COUNT(*) AS trip_count,
           ROUND(AVG(trip_distance), 2) AS avg_distance,
           ROUND(AVG(fare_amount), 2) AS avg_fare
    FROM taxi_trips
    GROUP BY passenger_count
    ORDER BY passenger_count
""")
q8.show()

# %%
# ──── SQL Query 9: Revenue by Vendor ────
print("📌 SQL Query 9: Revenue by Vendor")
q9 = spark.sql("""
    SELECT VendorID,
           COUNT(*) AS total_trips,
           ROUND(SUM(total_amount), 2) AS total_revenue,
           ROUND(AVG(total_amount), 2) AS avg_revenue_per_trip,
           ROUND(AVG(trip_distance), 2) AS avg_distance
    FROM taxi_trips
    GROUP BY VendorID
    ORDER BY VendorID
""")
q9.show()

# %%
# ──── SQL Query 10: Trips by Rate Code ────
print("📌 SQL Query 10: Trips by Rate Code")
q10 = spark.sql("""
    SELECT RatecodeID,
           CASE CAST(RatecodeID AS INT)
               WHEN 1 THEN 'Standard Rate'
               WHEN 2 THEN 'JFK'
               WHEN 3 THEN 'Newark'
               WHEN 4 THEN 'Nassau/Westchester'
               WHEN 5 THEN 'Negotiated Fare'
               WHEN 6 THEN 'Group Ride'
               ELSE 'Unknown'
           END AS rate_name,
           COUNT(*) AS trip_count,
           ROUND(AVG(fare_amount), 2) AS avg_fare,
           ROUND(AVG(trip_distance), 2) AS avg_distance
    FROM taxi_trips
    GROUP BY RatecodeID
    ORDER BY trip_count DESC
""")
q10.show()

# %%
# ──── Save Query Outputs to CSV ────
print("💾 Saving query results to output/ folder...")

q1.toPandas().to_csv("output/query1.csv", index=False)
print("   ✅ output/query1.csv — Top 10 longest trips")

q2.toPandas().to_csv("output/query2.csv", index=False)
print("   ✅ output/query2.csv — Top pickup locations")

q3.toPandas().to_csv("output/query3.csv", index=False)
print("   ✅ output/query3.csv — Average fare by payment type")

print("\n✅ All query outputs saved!")

# %% [markdown]
# ---
# ## Part 7: Window Functions (10 Marks)
# Using: row_number(), rank(), dense_rank()

# %%
# ============================================================
# PART 7: WINDOW FUNCTIONS (10 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/window_functions.png
#    → Take screenshot of all 3 window function outputs
# ============================================================

print("=" * 60)
print("        WINDOW FUNCTIONS")
print("=" * 60)

# Define window spec: partition by pickup location, order by fare descending
windowSpec = Window.partitionBy("PULocationID").orderBy(col("fare_amount").desc())

# %%
# ──── Window Function 1: row_number() ────
print("\n📌 Window Function 1: row_number()")
print("   Purpose: Assign unique sequential numbers to trips")
print("   within each pickup location, ordered by fare (highest first)\n")

df_rownum = df_clean.withColumn("row_num", row_number().over(windowSpec))
df_rownum.select("PULocationID", "fare_amount", "trip_distance", "row_num") \
    .filter(col("row_num") <= 3) \
    .orderBy("PULocationID", "row_num") \
    .show(15)

# %%
# ──── Window Function 2: rank() ────
print("\n📌 Window Function 2: rank()")
print("   Purpose: Rank trips by fare within each pickup location")
print("   (ties get the same rank, then ranks are skipped)\n")

df_rank = df_clean.withColumn("fare_rank", rank().over(windowSpec))
df_rank.select("PULocationID", "fare_amount", "trip_distance", "fare_rank") \
    .filter(col("fare_rank") <= 3) \
    .orderBy("PULocationID", "fare_rank") \
    .show(15)

# %%
# ──── Window Function 3: dense_rank() ────
print("\n📌 Window Function 3: dense_rank()")
print("   Purpose: Dense rank trips — no gaps in ranking for ties\n")

df_dense = df_clean.withColumn("dense_fare_rank", dense_rank().over(windowSpec))
df_dense.select("PULocationID", "fare_amount", "trip_distance", "dense_fare_rank") \
    .filter(col("dense_fare_rank") <= 3) \
    .orderBy("PULocationID", "dense_fare_rank") \
    .show(15)

# %% [markdown]
# ---
# ## Part 8: Performance Optimization (10 Marks)
# Demonstrate: cache(), repartition(), explain(), and execution time comparison

# %%
# ============================================================
# PART 8: PERFORMANCE OPTIMIZATION (10 Marks)
# ============================================================
# 📸 SCREENSHOT: screenshots/performance.png
#    → Take screenshot of timing comparison and explain() output
# ============================================================

print("=" * 60)
print("        PERFORMANCE OPTIMIZATION")
print("=" * 60)

# ──── Uncache if previously cached ────
df_clean.unpersist()
import time

# ──── Test WITHOUT cache ────
print("\n🔴 Test 1: WITHOUT Cache")
start_time = time.time()
result1 = df_clean.groupBy("PULocationID").agg(
    count("*").alias("trips"),
    avg("fare_amount").alias("avg_fare")
).collect()
no_cache_time = time.time() - start_time
print(f"   Execution Time: {no_cache_time:.4f} seconds")

# %%
# ──── Test WITH cache ────
print("\n🟢 Test 2: WITH Cache")
df_clean.cache()
# First action triggers the caching
df_clean.count()

start_time = time.time()
result2 = df_clean.groupBy("PULocationID").agg(
    count("*").alias("trips"),
    avg("fare_amount").alias("avg_fare")
).collect()
cache_time = time.time() - start_time
print(f"   Execution Time: {cache_time:.4f} seconds")

print(f"\n{'=' * 60}")
print(f"📊 PERFORMANCE COMPARISON:")
print(f"   Without Cache : {no_cache_time:.4f} seconds")
print(f"   With Cache    : {cache_time:.4f} seconds")
if cache_time > 0:
    print(f"   Speedup       : {no_cache_time/cache_time:.2f}x faster")
print(f"{'=' * 60}")

# %%
# ──── Repartition Demo ────
print("\n📌 Repartition Demonstration:")
print(f"   Current Partitions : {df_clean.rdd.getNumPartitions()}")
df_repart = df_clean.repartition(8)
print(f"   After repartition(8): {df_repart.rdd.getNumPartitions()}")
df_coalesce = df_clean.coalesce(2)
print(f"   After coalesce(2)   : {df_coalesce.rdd.getNumPartitions()}")

# %%
# ──── Execution Plan (explain) ────
# 📸 SCREENSHOT: screenshots/explain_plan.png
print("\n📌 Execution Plan — explain(True):")
print("   Query: GroupBy PULocationID → Average fare_amount\n")
df_clean.groupBy("PULocationID").agg(avg("fare_amount")).explain(True)

# %% [markdown]
# ---
# ## Part 9: Spark UI Analysis (5 Marks)
# 📸 Go to http://localhost:4040 and take these screenshots:
#    - screenshots/jobs.png
#    - screenshots/stages.png
#    - screenshots/storage.png
#    - screenshots/executors.png

# %%
# ============================================================
# PART 9: SPARK UI ANALYSIS (5 Marks)
# ============================================================
# 📸 SCREENSHOTS (from http://localhost:4040):
#    1. screenshots/jobs.png      → Click "Jobs" tab
#    2. screenshots/stages.png    → Click "Stages" tab
#    3. screenshots/storage.png   → Click "Storage" tab
#    4. screenshots/executors.png → Click "Executors" tab
# ============================================================

print("=" * 60)
print("        SPARK UI ANALYSIS")
print("=" * 60)
print("""
🌐 Open your browser and go to: http://localhost:4040

Take the following screenshots:

1. JOBS tab       → Save as: screenshots/jobs.png
2. STAGES tab     → Save as: screenshots/stages.png
3. STORAGE tab    → Save as: screenshots/storage.png
4. EXECUTORS tab  → Save as: screenshots/executors.png

After taking screenshots, answer these questions:
""")

# Print some metrics to help answer Spark UI questions
print(f"📊 Spark UI Analysis Answers:")
print(f"   • Total Jobs executed    : Check the Jobs tab for exact count")
print(f"   • Total Stages           : Check the Stages tab for exact count")
print(f"   • Shuffle operation       : groupBy() causes shuffle")
print(f"   • Why shuffle occurs      : groupBy requires redistributing data")
print(f"     across partitions so all records with the same key")
print(f"     end up in the same partition for aggregation.")
print(f"\n⚠️  Keep this notebook running while you take the screenshots!")
print(f"    The Spark UI is only available while the SparkSession is active.")

# %%
# ============================================================
# STOP HERE — Take your Spark UI screenshots before running
# the cell below! The cell below stops Spark.
# ============================================================
input("⏸️  Press ENTER after you have taken all Spark UI screenshots...")

# %%
# ──── Stop Spark Session ────
print("\n🛑 Stopping Spark Session...")
spark.stop()
print("✅ Spark Session stopped successfully.")
print("\n🎉 Assignment Complete! All parts executed successfully.")
