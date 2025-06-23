# Dataproc Overview

## Introduction
- Dataproc provides fast, easy, cost-effective way to run Apache Hadoop and Apache Spark
- Apache Hadoop and Apache Spark are open source technologies that form the foundation of big data processing

### Apache Hadoop
- Set of tools and technologies for cluster computing
- Enables storage and processing of large data volumes
- Intelligently connects individual computers together in a cluster to distribute storage and processing of data

### Apache Spark
- Unified analytics engine for large-scale data processing
- Achieves high performance for both batch and stream data

### Dataproc Definition
- Managed Spark and Hadoop service
- Enables use of open source data tools for:
  - Batch processing
  - Querying
  - Streaming
  - Machine learning
- Features automated cluster management
- Clusters are typically run ephemerally (turned off when not needed) for cost savings

## Key Features of Dataproc

### Cost Effectiveness
- Priced at 1¢ per virtual CPU per cluster per hour
- Additional Google Cloud resource costs apply
- Supports preemptible instances for lower compute prices
- Pay-as-you-need model: use and pay only when needed

### Speed and Scalability
- Quick cluster operations (90 seconds or less on average):
  - Start
  - Scale
  - Shutdown
- Flexible configuration options:
  - Multiple virtual machine types
  - Various disk sizes
  - Adjustable number of nodes
  - Networking options

### Open Source Ecosystem
- Compatible with existing tools and documentation:
  - Spark
  - Hadoop
  - Pig
  - Hive
- Frequent updates to native versions
- No need to learn new tools or APIs
- Existing projects or ETL pipelines can be moved without redevelopment

### Management Features
- Easy interaction without administrator or special software through:
  - Google Cloud Console
  - Google Cloud SDK
  - Dataproc REST API
- Clusters can be turned off when not needed to prevent idle costs

### Additional Features
- Image versioning: Switch between different versions of:
  - Apache Spark
  - Apache Hadoop
  - Other tools
- Built-in integrations with:
  - Cloud Storage
  - BigQuery
  - Cloud Bigtable
  - Cloud Logging
  - Cloud Monitoring
- Provides complete data platform beyond basic Spark/Hadoop cluster
- Can process terabytes of raw log data directly into BigQuery for business reporting

## Use Cases

### Case 1: Daily Log Processing
**Scenario:**
- 50GB daily text log processing from multiple sources
- Need for aggregated data for:
  - Daily reporting
  - Management dashboards
  - Analysis
- Currently using dedicated on-premises cluster with MapReduce

**Solution:**
1. Cloud Storage as low-cost landing zone for log data
2. Create Dataproc cluster in under 2 minutes
3. Process using existing MapReduce
4. Remove cluster immediately after completion

**Benefits:**
- Cost reduction through ephemeral cluster usage
- No costs during idle periods
- Reduced complexity

### Case 2: Spark Shell Analytics
**Scenario:**
- Analysts rely on and are comfortable using Spark Shell
- IT department concerned about:
  - Increasing usage
  - Scaling challenges with standalone cluster mode

**Solution:**
- Scalable Dataproc clusters to handle increased demand
- Supports multiple Spark technologies:
  - Spark
  - Spark SQL
  - PySpark
- Multiple interface options:
  - Web interface
  - Cloud SDK
  - Native Spark Shell through SSH

**Benefits:**
- Unlocks cloud power without added technical complexity
- Reduces computation time from minutes/hours to seconds
- Maintains familiar workflow for analysts

### Case 3: Machine Learning Implementation
**Scenario:**
- Uses Spark machine learning libraries (MLlib)
- Runs classification algorithms on very large datasets
- Currently using cloud-based machines with custom Spark installation

**Solution:**
- Quick Dataproc cluster creation with pre-installed Spark and MLlib
- Custom cluster configurations via initialization actions
- Workflow monitoring through:
  - Cloud Logging
  - Cloud Monitoring

**Benefits:**
- Focus on data rather than cluster management
- Easy customization across entire cluster
- Access to new Google Cloud product integrations
- Enhanced features for Spark clusters

# Introduction to Dataflow

## Overview
- Service for ETL (Extract, Transform, Load) operations
- Alternative to Dataproc for non-Hadoop dependent processing
- Handles both batch and streaming data processing

## Core Features
- Fully managed service by Google Cloud
- Optimized for:
  - Large-scale batch processing
  - Long-running stream processing
- Creates pipelines for data processing

## Pipeline Design Considerations

### Key Questions
1. Data Compatibility:
   - Will code work with both batch and streaming?
   - Need for code refactoring?

2. SDK Requirements:
   - Availability of necessary transformations
   - Mid-flight aggregations
   - Windowing capabilities

3. Additional Considerations:
   - Late data handling capability
   - Availability of existing templates/solutions

## Operational Features

### Automation
- Automated operational tasks:
  - Resource management
  - Performance optimization
- On-demand resource provisioning
- Automatic scaling based on requirements

### Reliability
- Built-in fault-tolerant execution
- Consistent and correct processing regardless of:
  - Data size
  - Cluster size
  - Processing pattern
  - Pipeline complexity

### Monitoring and Integration
- Real-time monitoring through Google Cloud Console:
  - Pipeline throughput statistics
  - Lag metrics
  - Consolidated worker log inspection

### Platform Integration
Seamless integration with:
- Cloud Storage
- Pub/Sub
- Datastore
- Cloud Bigtable
- BigQuery


# BigQuery Overview

## Introduction
- Google's fully managed, petabyte-scale, low-cost analytics data warehouse
- Serverless solution that handles infrastructure management
- Data warehouse stores terabytes/petabytes of data from wide range of organizational sources
- Designed to guide management decisions
- Allows focus on SQL queries for business questions without infrastructure concerns

## Key Features

### Dual Functionality
1. **Storage**
   - Petabyte-scale capacity
   - Context: 1 petabyte equals 11,000 movies at 4K quality
   - Fully managed storage solution

2. **Analytics**
   - Built-in capabilities:
     - Machine learning
     - Geospatial analysis
     - Business intelligence
   - Advanced analytical features for comprehensive data analysis

### Management and Infrastructure
- Fully managed serverless solution
- SQL-based querying interface
- No backend infrastructure management required
- Resources for SQL beginners provided through course materials and labs
- Complete abstraction of infrastructure complexity

### Pricing Models
1. **Pay-as-you-go**
   - Charges based on:
     - Number of bytes processed in queries
     - Permanent table storage costs
   - Flexible pricing for varying usage patterns

2. **Flat-rate**
   - Fixed monthly billing option
   - Reserved resource allocation
   - Predictable cost structure

### Security
- Automatic encryption at rest by default
- No customer action required for encryption
- Comprehensive protection for:
  - Stored disk data
  - Solid-state drives
  - Backup media
  - All data storage systems

### Machine Learning Integration
- Built-in ML features for SQL-based model creation
- Direct integration with Vertex AI
- Seamless dataset export to Vertex AI
- Complete data-to-AI lifecycle support
- Accessible to users without extensive ML expertise

## Architecture

### Data Input Types
1. **Real-time (Streaming) Data**
   - Handles structured and unstructured data
   - Processes high-speed, large volume inputs
   - Utilizes Pub/Sub for data ingestion
   - Ensures real-time data processing

2. **Batch Data**
   - Direct upload capability to Cloud Storage
   - Efficient bulk data processing

### Data Processing Flow
- Dataflow handles ETL operations
- BigQuery central position in architecture:
  - Connects data processing systems
  - Links to analytics tools
  - Facilitates AI/ML integration
  - Manages data pipeline connections

### Output Integration

1. **Business Intelligence Tools**
   - Multiple platform support:
     - Looker
     - Looker Studio
     - Tableau
   - Google Sheets integration features:
     - Direct dataset querying
     - Support for small and large datasets
     - Built-in pivot table functionality
     - Spreadsheet operations support

2. **AI/ML Tools**
   - Direct integration with:
     - AutoML
     - Vertex AI Workbench
   - Part of comprehensive Vertex AI platform
   - Seamless data scientist workflow support

## Data Sources and Management

### Input Sources
1. **Internal Data**
   - Native BigQuery storage
   - Directly managed datasets

2. **External Data**
   - Multiple source support:
     - Cloud Storage
     - Spanner
     - Cloud SQL
     - Raw CSV files
     - Google Sheets
   - Direct querying without prior ingestion

3. **Multi-cloud Data**
   - AWS integration
   - Azure integration
   - Cross-platform compatibility

4. **Public Datasets**
   - Available through Cloud Marketplace
   - Ready-to-use data collections

### Data Management
- Automatic replication
- Built-in backup systems
- Autoscaling capability
- Full lifecycle management

### Data Loading Patterns
1. **Batch Load**
   - Single operation processing
   - Options for:
     - One-time operations
     - Scheduled operations
   - Table management:
     - New table creation
     - Existing table appendment

2. **Streaming**
   - Continuous small batch processing
   - Near real-time data availability
   - Immediate query access

3. **Generated Data**
   - SQL-based data insertion
   - Query result storage
   - Dynamic data generation

## Analytics Capabilities

### Performance Metrics
- Terabyte-scale processing in seconds
- Petabyte-scale processing in minutes
- Real-time insight generation
- Optimized for large-scale analysis

### Machine Learning Features
- Built-in ML modeling capabilities
- Designed for data analyst accessibility
- SQL-based development environment
- Reduced need for:
  - Extensive programming knowledge
  - ML framework expertise
  - Custom coding

### Access Methods
1. **BigQuery Web UI**
   - Interactive interface
   - Visual query builder

2. **bq Command-line Tool**
   - Terminal-based access
   - Automation capabilities

3. **BigQuery REST API**
   - Programmatic access
   - Integration support

4. **External Tools**
   - Jupyter notebook support
   - BI platform integration
   - Custom tool connectivity

### Role-Based Access
- Supports multiple user types:
  - Business analysts
  - BI developers
  - Data scientists
  - ML engineers
- Customizable access permissions
- Common staging area for all data analytics workloads

