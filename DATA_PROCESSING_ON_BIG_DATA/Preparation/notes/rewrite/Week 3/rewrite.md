# Week 3: GCCF: Infrastructure in Google Cloud Part 2, & Data, ML, AI in Google Cloud

## Introduction to Big Data Managed Services in the Cloud

In this first section, we'll discuss big data managed services in the cloud. Before we explore this in detail, let's take a moment to conceptualize big data. Enterprise storage systems are leaving the terabyte behind as a measure of data size, with petabytes becoming the norm. We know that one petabyte is 1,000,000 GB or 1,000 TB. But how big is that?

From one perspective, a petabyte of data might seem like more than you'll ever need. For example, you need a stack of floppy disks higher than 12 Empire State buildings to store one petabyte. If you wanted to download one petabyte over a 4G network, you'd have to sit and wait for 27 years. You'd also need one petabyte of storage for every tweet ever tweeted, multiplied by 50. So one petabyte is pretty big.

If we look at it from a different perspective, though, one petabyte is only enough to store two micrograms of DNA or one day's worth of video uploaded to YouTube. So for some industries, a petabyte of data might not be much at all. Every company stores data in some way, and now they're trying to use that data to gain some insight into their business operations.

This is where big data comes in. Big data architectures allow companies to analyze their stored data to learn about their business. In this module, we'll focus on three managed services that Google offers for the processing of data.

For companies that have already invested in Apache Hadoop and Apache Spark and want to continue using these tools, Dataproc provides a great way to run open source software in Google Cloud. However, companies looking for a streaming data solution might be more interested in Dataflow as a managed service. Dataflow is optimized for large scale batch processing or long-running streaming, processing of structured and unstructured data.

The third managed service that we'll look at is BigQuery, which provides a data analytics solution optimized for getting answers rapidly over petabyte scale datasets. BigQuery allows for fast SQL unstructured data.

## Leverage Big Data Operations with Dataproc

In this section, we'll learn how Dataproc provides a fast, easy, cost-effective way to run Apache Hadoop and Apache Spark. Apache Hadoop and Apache Spark are open source technologies that often are the foundation of big data processing. Apache Hadoop is a set of tools and technologies which enables a cluster of computers to store and process large volumes of data. It intelligently ties individual computers together in a cluster to distribute the storage and processing of data.

Apache Spark is a unified analytics engine for large-scale data processing and achieves high performance for both batch and stream data. Dataproc is a managed Spark and Hadoop service that lets you use open source data tools for batch processing, querying, streaming, and machine learning. Dataproc automation helps you create clusters quickly, manage them easily, and because clusters are typically run ephemerally, you save money as they are turned off when you don't need them.

Let's look at the key features of Dataproc. Cost effective: Dataproc is priced at 1 cent per virtual CPU per cluster per hour, on top of any other Google Cloud resources you use. In addition, Dataproc clusters can include preemptible instances that have lower compute prices. You use and pay for things only when you need them.

Fast and scalable: Dataproc clusters are quick to start, scale, and shut down, and each of these operations takes 90 seconds or less, on average. Clusters can be created and scaled quickly with many virtual machine types, disk sizes, number of nodes, and networking options.

Open source ecosystem: You can use Spark and Hadoop tools, libraries, and documentation with Dataproc. Dataproc provides frequent updates to native versions of Spark, Hadoop, Pig, and Hive, so learning new tools or APIs is not necessary, and you can move existing projects or ETL pipelines without redevelopment.

Fully managed: You can easily interact with clusters and Spark or Hadoop jobs, without the assistance of an administrator or special software, through the Google Cloud console, the Google Cloud SDK, or the Dataproc REST API. When you're done with a cluster, simply turn it off, so money isn't spent on an idle cluster.

Image versioning: Dataproc's image versioning feature lets you switch between different versions of Apache Spark, Apache Hadoop, and other tools. Built-in integration: The built-in integration with Cloud Storage, BigQuery, and Cloud Bigtable ensures that data will not be lost. This, together with Cloud Logging and Cloud Monitoring, provides a complete data platform and not just a Spark or Hadoop cluster. For example, you can use Dataproc to effortlessly extract, transform, and load terabytes of raw log data directly into BigQuery for business reporting.

Let's look at a few Dataproc use cases. In this first example, a customer processes 50 gigabytes of text log data per day from several sources. The objective is to produce aggregated data that is then loaded into databases from which metrics are gathered for daily reporting, management dashboards, and analysis. Until now, they have used a dedicated on-premises cluster to store and process the logs with MapReduce.

So what's the solution? First, Cloud Storage can act as a landing zone for the log data at a low cost. A Dataproc cluster can then be created in less than 2 minutes to process this data with their existing MapReduce. Once completed, the Dataproc cluster can be removed immediately. In terms of value, instead of running all the time and incurring costs even when not used, Dataproc only runs to process the logs, which reduces cost and complexity.

Now, let's analyze a second example. In this organization, analysts rely on—and are comfortable using—Spark Shell. However, their IT department is concerned about the increase in usage, and how to scale their cluster, which is running in standalone mode. The solution is for Dataproc to create clusters that scale for speed and mitigate any single point of failure. Since Dataproc supports Spark, Spark SQL, and PySpark, they could use the web interface, Cloud SDK, or the native Spark Shell through SSH. The value is Dataproc's ability to quickly unlock the power of the cloud for anyone without added technical complexity. Running complex computations would take seconds instead of minutes or hours.

In this third example, a customer uses the Spark machine learning libraries (MLlib) to run classification algorithms on very large datasets. They rely on cloud-based machines where they install and customize Spark. Because Spark and the MLlib can be installed on any Dataproc cluster, the customer can save time by quickly creating Dataproc clusters. Any additional customizations can be applied easily to the entire cluster through initialization actions. To monitor workflows, they can use the built-in Cloud Logging and Cloud Monitoring. In terms of value, resources can be focused on the data with Dataproc, not spent on cluster creation and management. Integrations with new Google Cloud products also unlock new features for Spark clusters.

## Build Extract, Transform, and Load Pipelines Using Dataflow

In this section, we'll learn how you can use Dataflow to perform extract, transform, and load operations.

With Dataproc, you can migrate your original big data deployment with Apache Hadoop and Apache Spark to a fully-managed service provided by Google Cloud.

But how do you process both batch and streaming data if it's not Hadoop dependent? This is where Dataflow comes in.

Dataflow is a managed service offered by Google that's optimized for large-scale batch processing or long-running stream processing. Dataflow creates a pipeline to process both streaming data and batch data. "Process" in this case refers to the steps to extract, transform, and load data (ETL).

When building a data pipeline, data engineers often encounter challenges related to coding the pipeline design and implementing and serving the pipeline at scale.

During the pipeline design phase, you should consider a few questions: Will the pipeline code be compatible with both batch and streaming data, or will it need to be refactored? Will the pipeline code software development kit, or SDK, that's being used have all the transformations, mid-flight aggregations and windowing? Will it be able to handle late data? Are there existing templates or solutions that should be referenced?

Dataflow fully automates operational tasks like resource management and performance optimization. All resources are provided on demand, and scale to meet requirements.

Dataflow provides built-in support for fault-tolerant execution that's consistent and correct regardless of data size, cluster size, processing pattern or pipeline complexity.

Through its integration with the Google Cloud console, Dataflow provides statistics such as pipeline throughput and lag and consolidated worker log inspection—all in near-real time. It also integrates with Cloud Storage, Pub/Sub, Datastore, Cloud Bigtable, and BigQuery for seamless data processing between platforms.

## BigQuery, Google's Enterprise Data Warehouse

In this last section, we'll learn about BigQuery, which is Google's fully managed, petabyte-scale, low-cost analytics data warehouse. BigQuery is a fully-managed, serverless data warehouse. A data warehouse is a large store that contains terabytes and petabytes of data gathered from a wide range of sources within an organization, and it's used to guide management decisions.

Being fully managed means that BigQuery takes care of the underlying infrastructure, so you can focus on using SQL queries to answer business questions without worrying about deployment, scalability, and security.

Let's look at some key features of BigQuery. BigQuery provides two services in one: storage plus analytics. It's a place to store petabytes of data. For reference, 1 petabyte is equivalent to 11,000 movies at 4k quality. BigQuery is also a place to analyze data, with built-in features like machine learning, geospatial analysis, and business intelligence, which we'll explore a bit later on.

BigQuery is a fully managed serverless solution, which means that you use SQL queries to answer your organization's biggest questions in the frontend without worrying about infrastructure in the backend. If you've never written SQL before, don't worry. This course provides resources and labs to help.

BigQuery has a flexible pay-as-you-go pricing model where you pay for the number of bytes of data your query processes and for any permanent table storage. If you prefer to have a fixed bill every month, you can also subscribe to flat-rate pricing where you have a reserved amount of resources for use.

Data in BigQuery is encrypted at rest by default without any action required from a customer. By encryption at rest, we mean that encryption is used to protect data that is stored on a disk, including solid-state drives or backup media.

BigQuery has built-in machine learning features, so you can write ML models directly in BigQuery by using SQL. Also, if you use other professional tools—such as Vertex AI from Google Cloud—to train your ML models, you can export datasets from BigQuery directly into Vertex AI for a seamless integration across the data-to-AI lifecycle.

So what does the typical architecture of a data warehouse solution look like? The input data can be either real-time or batch data. If it's streaming data, which can be either structured or unstructured, high speed, and large volume, Pub/Sub is needed to digest the data. If it's batch data, it can be directly uploaded to Cloud Storage.

After that, both pipelines lead to Dataflow to process the data. Dataflow is where we extract, transform, and load the data if needed. BigQuery sits in the middle to link data processes by using Dataflow and data access through analytics, AI, and ML tools.

The job of the analytics engine of BigQuery at the end of a data pipeline is to ingest all the processed data after ETL, store and analyze it, and then possibly output it for further use such as data visualization and machine learning.

BigQuery outputs usually feed into two buckets: business intelligence tools and AI/ML tools. If you're a business analyst or data analyst, you can connect to visualization tools like Looker, Looker Studio, Tableau, and other BI tools. If you prefer to work in spreadsheets, you can query both small or large BigQuery datasets directly from Google Sheets and even perform common operations like pivot tables.

Alternatively if you're a data scientist or machine learning engineer, you can directly call the data from BigQuery through AutoML or Vertex AI Workbench. These AI/ML tools are part of Vertex AI, Google's unified ML platform.

BigQuery is like a common staging area for data analytics workloads. When your data is there, business analysts, BI developers, data scientists, and machine learning engineers can be granted access to your data for their own insights.

BigQuery can ingest datasets from various sources, including Internal data, which is data saved directly in BigQuery, External data. BigQuery also offers the option to query external data sources like data stored in other Google Cloud storage services such as Cloud Storage, or in other Google Cloud database services, such as Spanner or Cloud SQL and bypass BigQuery managed storage.

This means that a raw CSV file in Cloud Storage or a Google Sheet can be used to write a query without being ingested by BigQuery first. Multi-cloud data, which is data stored in multiple cloud services, such as AWS or Azure, and Public datasets. If you don't have data of your own, you can analyze any of the public datasets available in the Cloud Marketplace.

After the data is stored in BigQuery, it's fully managed and is automatically replicated, backed up, and set up to autoscale. You can use three basic patterns to load data into BigQuery: batch load, where source data is loaded into a BigQuery table in a single batch operation. This can be a one-time operation or it can be automated to occur on a schedule. A batch load operation can create a new table or append data into an existing table.

The second is streaming, where smaller batches of data are streamed continuously so that the data is available for querying in near-real time. And the third is generated data, where SQL statements are used to insert rows into an existing table or to write the results of a query to a table.

Of course, the purpose of BigQuery is not to just store data; it's for analyzing data and helping to make business decisions. BigQuery is optimized for running analytic queries over large datasets. It can perform queries on terabytes of data in seconds and petabytes in minutes. This performance lets you analyze large datasets efficiently and get insights in near real time.

Looker, Looker Studio, and many integrated partner tools can be used to draw analytics from BigQuery and build sophisticated interactive data visualizations. BigQuery also has built-in capabilities for building machine learning models. An ML model lets you solve certain kinds of problems at scale by using data examples, but without the need for custom code.

Machine learning on large datasets requires extensive programming and knowledge of ML frameworks. These requirements restrict solution development to a small set of people within each company, and they exclude data analysts who understand the data but have limited machine learning knowledge and programming expertise.

BigQuery ML empowers data analysts to use machine learning through existing SQL tools and skills. Analysts can use BigQuery ML to build and evaluate ML models in BigQuery. Analysts no longer need to export small amounts of data to spreadsheets or other applications, and they no longer need to wait for limited resources from a data science team.

BigQuery ML functionality is available by using the BigQuery web UI, the bq command-line tool, the BigQuery REST API, and an external tool such as a Jupyter notebook or business intelligence platform.
