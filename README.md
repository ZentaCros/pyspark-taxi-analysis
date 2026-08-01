# 🚕 NYC Taxi Trip Analytics using Apache Spark

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.5-orange.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A big data processing pipeline built with **PySpark** to analyze millions of New York City taxi trips. This project demonstrates distributed data processing, functional transformations, SQL-based declarative analytics, and performance optimization techniques using Apache Spark.

---

## 📌 Features

- **Data Ingestion & Cleaning:** Processing raw `.parquet` files from the official NYC TLC dataset.
- **Spark Transformations:** Utilizing PySpark DataFrame APIs (`filter`, `select`, `withColumn`, `groupBy`, `join`, `repartition`) for feature engineering and aggregation.
- **Declarative Analytics (Spark SQL):** Executing ANSI SQL queries against distributed DataFrames to uncover business insights (e.g., peak hours, longest trips, revenue by vendor).
- **Advanced Analytics:** Implementing Window Functions (`row_number`, `rank`, `dense_rank`) for partitioned ranking (e.g., top-revenue trips per pickup location).
- **Performance Optimization:** Benchmarking Spark's in-memory caching (`.cache()`) and analyzing physical execution plans (`explain()`).
- **Cluster Monitoring:** Tracking job execution, stage boundaries, shuffle reads/writes, and DAG visualization via the Spark Web UI.

---

## 📊 Dataset

- **Source:** [NYC Taxi and Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Data File:** Yellow Taxi Trip Records — January 2024
- **Format:** Apache Parquet
- **Size:** ~3 million records / ~50 MB compressed

---

## 🛠️ Architecture & Workflow

1. **Initialization:** A SparkSession is established with local mode execution (`local[*]`).
2. **Load & Schema Inference:** Parquet metadata is used to infer the dataframe schema without scanning the entire file.
3. **Data Quality Pipeline:** Removal of negative fares, zero-distance trips, and nulls in critical columns.
4. **Feature Engineering:** Calculation of trip durations (minutes) and cost-per-mile metrics.
5. **Analytics Execution:** Running 10 distinct SQL queries and window function aggregations.
6. **Result Extraction:** Collecting distributed results back to the driver node for reporting.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Java 8 or 11 (required for Spark)
- `pip` package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ZentaCros/pyspark-taxi-analysis.git
   cd pyspark-taxi-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Dataset:
   Download the [January 2024 Yellow Taxi Parquet file](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet) and place it in the project root directory.

### Execution

Run the Python script directly:
```bash
python src/nyc_taxi_spark.py
```

Or open the Jupyter Notebook for interactive execution:
```bash
jupyter notebook nyc_taxi_analysis.ipynb
```

---

## 📈 Key Findings

- **Caching Performance:** Implementing `.cache()` before heavy aggregations yielded an **11.68x speedup** (execution time dropped from ~10.5 seconds to ~0.9 seconds).
- **Peak Demand:** Pickup volume peaks consistently between 5:00 PM and 7:00 PM (rush hour).
- **Vendor Dominance:** Creative Mobile Technologies (CMT) handled a significantly larger volume of the highest-revenue long-distance trips compared to VeriFone Inc.
- **Payment Trends:** Credit cards accounted for the vast majority of payments, consistently driving higher average tips compared to cash.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
*Developed by Muhammad Hamza Azeem (221980023) as part of the Big Data Analytics (DS-313) coursework at GIFT University.*
