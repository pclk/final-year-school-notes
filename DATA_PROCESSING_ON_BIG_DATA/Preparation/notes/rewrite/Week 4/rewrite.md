# Week 4: Data Engineering - Building Batch Data Pipelines in Google Cloud Part 1

## EL, ELT, ETL

Let's start with a quick recap of EL, ELT and ETL.

EL is Extract and Load. This refers to when data can be imported as is into a system.

ELT or Extract, Load and Transform, allows raw data to be loaded directly into the target and transformed whenever it is needed. For example, you might provide access to the raw data through a view that determines whether the user wants all transactions or only reconciled ones.

ETL or Extract, Transform and Load, is a data integration process in which transformation takes place in an intermediate service before it is loaded into the target. For example, the data might be transformed in Dataflow before being loaded into BigQuery.

When would you use EL? The bottom line is that you should use EL only if the data is already clean and correct. Perhaps you have log files in Cloud Storage. You can extract data from files on Cloud Storage and load it into BigQuery's native storage. This is a simple REST API call.

You can trigger this pipeline from Cloud Composer, Cloud Functions or via a scheduled query. You might even set it to work in micro batches, not quite streaming but near real time. Whenever a new file hits Cloud Storage, the Cloud Function runs, and the function invokes a BigQuery job. The data transfer service in BigQuery will also work here.

Use EL for batch loading historical data or to do scheduled loads of log files. But let me emphasize, use EL only if the data is already clean and correct.

ELT starts with EL, so the loading is the same and could work the same way. File hits Cloud Storage, function invokes BigQuery load, table appended to. The big difference is what happens next. The table might be stored in a private dataset and everyone accesses the data through a view which imposes data integrity checks. Or maybe you have a job that runs a SQL query with a destination table. This way, transformed data is stored in a table that everyone accesses.

When do you use ELT? One common case is when you don't know what kind of transformations are needed to make the data usable. For example, let's say someone uploads a new image. You invoke the Vision API and back comes a long JSON message about all kinds of things in the image, text in the image, whether there's a landmark, a logo, what objects.

What will an analyst need in the future? You don't know, so you store the raw JSON as is. Later, if someone wants to count the number of times a specific company's logos are in this set of images, they can extract logos from the JSON and then count them.

Of course, this works only if the transformation that's needed can be expressed in SQL. In the case of the Vision API, the result is JSON and BigQuery SQL has support for JSON parsing, so ELT will work in this case.

## Quality Considerations

Now that we have looked at EL and ELT, let's look at some of the transformations you might want to do and how they can be done in BigQuery.

To keep things precise, let's assume that our data processing needs all revolve around quality improvements.

What are some of the quality related reasons why we might want to process data? The top row are characteristics of information. Information can be valid, accurate, complete, consistent and or uniform. These terms are defined in the science of logic, each is independent. For example, data can be complete without being consistent, it can be valid without being uniform.

There are formal definitions for each of these terms that you can look up online. But the main practical reason for seeking them is shown in the second row, the problems they present in data analysis. It is one thing to seek each of the five badges for your data to have objectively good data quality. However, it is another thing when poor quality data interferes with data analysis and leads to incorrect business decisions.

So the reason to spend time, energy, and resources detecting and resolving quality issues is that it can affect a business outcome. Thus, if data does not conform to your business rules, you have a problem of validity. For example, let's say that you sell movie tickets, and each ticket costs $10. If you have a $7 transaction, then you have a validity problem.

Similarly, accuracy problems are due to data not conforming to objective truth. Completeness has to do with failing to process everything. Consistency problems are if two different operations ought to be the same but yield different results, and because you don't know what to trust, you can't derive insights from the data. Uniformity is when data values of the same column in different rows mean different things.

The main causes of these problems are listed in the third row. We will explore methods of detecting each of these issues in data. Now you have found the problems, what do you do about them? ELT and BigQuery can often help fix many data quality issues.

Here is an example. Imagine you plan to analyze data but there are duplicate records making it seem like one kind of event is more common, when in fact this is just a data quality issue. You cannot derive insights from the data until the duplicates are removed.

So do you need a transformation step to remove the duplicates before you store the data? Maybe, but a simpler solution exists, to count unique records. You do, of course, have count distinct in BigQuery and you can use that instead.

Similarly, a problem like data being out of range can be solved in BigQuery without an intermediate transformation step. Invalid data can be filtered out using a BigQuery view and everyone can access the view rather than the raw data.

## How to carry out operations in BigQuery

In this lesson, we will look at various quality issues and talk through some BigQuery capabilities that can help you address those quality problems. We can use Views to filter out rows that have quality issues. For example, remove quantities less than zero using a WHERE clause. After you do a GROUP BY, you can discard groups whose total number of records is less than 10 using the HAVING clause.

Think carefully about how you wish to treat nulls and blanks. A NULL is the absence of data. A BLANK is an empty string. Consider if you are trying to filter out both NULLS and BLANKS or only NULLs or only BLANKs. You can easily count non-null values using COUNTIF and use the IF statement to avoid using specific values in computations.

For accuracy, test data against known good values. For example, if you have an order, you could compute the sub_total from the quantity_ordered and item_price and make sure the math is accurate. Similarly, you can check if a value that is being inserted belongs to a canonical list of acceptable values. You can do that with a SQL IN.

For completeness, identify any missing values and either filter out, or replace them with something reasonable. If the missing value is NULL, SQL provides functions like NULLIF, COUNTIF, COALESCE, etc. to filter missing values out of calculations. You might be able to do a UNION from another source to account for missing months of data. The automatic process of detecting data drops and requesting data items to fill in the gaps is called "backfilling". It is a feature of some data transfer services.

When loading data, verify file integrity with checksum values (hash, MD5). Consistency problems are often due to duplicates. You expect that something is unique, and it isn't, so things like totals are wrong. COUNT provides the number of rows in a table that contain a non-null value. COUNT DISTINCT provides the number of unique values. If they are different, then it means that you have duplicate values. Similarly, if you do a GROUP BY, and any group contains more than one row, then you know you have two or more occurences of that value.

Another reason that you might have consistency problems is if extra characters have been added to the fields. For example, you may be getting timestamps, some of which may include a timezone. Or you have strings that are padded. Use string functions to clean such data before passing it on.

What happens if you are storing some value in centimeters, and suddenly, you start getting the value in millimeters? Your data warehouse will end up with non-uniform data. You have to safeguard against this. Use SQL cast to avoid issues with data types changing within a table. Use the SQL FORMAT() function to clearly indicate units. And in general, document them very clearly.

I hope that what you are coming away with is the idea that BigQuery SQL is very powerful and you can take advantage of this.

## Shortcomings

In the previous lesson, we showed you some of the ways in which you can use SQL in an ELT pipeline to safeguard against quality issues. The point is that you don't always need ETL. ELT might be an option even if you need transformation. However, there are situations where ELT won't be enough. In that case, ETL might be what you need to do.

What are the kinds of situations where it is appropriate? The first example - translating Spanish to English - requires calling an external API. This cannot be done directly in SQL. It is possible to use a BigQuery remote function, invoke the Cloud Translation API, and perform content translation. But this involves programming outside of BigQuery.

The second example - looking at a stream of customer actions over a time window - is rather complex. You can do it with windowed aggregations, but it is far simpler with programmatic logic. So, if the transformations cannot be expressed in SQL or are too complex to do in SQL, you might want to transform the data before loading it into BigQuery.

The reference architecture for Google Cloud suggests Dataflow as an ETL tool. We recommend that you build ETL pipelines in Dataflow and land the data in BigQuery. The architecture looks like this: Extract data from Pub/Sub, Cloud Storage, Spanner, Cloud SQL, etc. Transform the data using Dataflow, and have the Dataflow pipeline write to BigQuery.

When would you do this? When the raw data needs to be quality-controlled, transformed, or enriched before being loaded into BigQuery, and the transforms are difficult to do in SQL. When the data loading has to happen continuously, i.e. if the use case requires streaming. Dataflow supports streaming. We'll look at streaming in more detail in the next course.

And when you want to integrate with continuous integration / continuous delivery (CI/CD) systems and perform unit testing on all components. It's easy to schedule the launch of a Dataflow pipeline.

Dataflow is not the only option you have on Google Cloud if you want to do ETL. In this course, we will look at several data processing and transformation services that Google Cloud provides: Dataflow, Dataproc, and Data Fusion.

Dataproc and Dataflow can be used for more complex ETL pipelines. Dataproc is based on Apache Hadoop and requires significant Hadoop expertise to leverage directly. Data Fusion provides a simple graphical interface to build ETL pipelines that can then be easily deployed at scale to Dataproc clusters.

Dataflow is a fully managed, serverless data processing service based on Apache Beam that supports both batch and streaming data processing pipelines. While significant Apache Beam expertise is desirable in order to leverage the full power of Dataflow, Google also provides quick-start templates for Dataflow to allow you to rapidly deploy a number of useful data pipelines.

You can use any of these three products to carry out data transformation and then store the data in a data lake or data warehouse to support advanced analytics.

## ETL to solve data quality issues

Let's look at using ETL to solve data quality issues. Unless you have specific needs, we recommend that you use Dataflow and BigQuery.

What are a few needs that cannot be met easily with Dataflow and BigQuery? Low latency and high throughput. BigQuery queries are subject to a latency on the order of a few hundred milliseconds and you can stream on the order of a million rows per second into a BigQuery table -- this used to be 100,000 rows, but recently it got raised to 1 million per project. The typical latency number quoted for BigQuery is on the order of a second, but with BI engine it is possible to get latency on the order of 100 milliseconds -- you should always check the documentation and the solutions pages for the latest values.

If your latency and throughput considerations are more stringent, then Bigtable might be a better sink for your data processing pipelines.

Reusing Spark pipelines. Maybe you already have a significant investment in Hadoop and Spark. In that case, you might be a lot more productive in a familiar technology. Use Spark if that's what you know really well.

Need for visual pipeline building. Dataflow requires you to code data pipelines in Java or Python. If you want to have data analysts and non-technical users create data pipelines, use Cloud Data Fusion. They can drag-and-drop and visually build pipelines.

We'll look at all these options briefly now and in greater detail in the remainder of this course. Dataproc is a managed service for batch processing, querying, streaming, and Machine Learning. It is a service for Hadoop workloads and is quite cost effective when taking into consideration eliminating the tasks related to running Hadoop on bare metal and taking on all of the related maintenance activities. It also has a few powerful features like auto-scaling and out-of-the-box integration with Google Cloud products like BigQuery.

Cloud Data Fusion is a fully-managed, cloud-native, enterprise data integration service for quickly building and managing data pipelines. You can use it to populate a data warehouse, but you can also use it for transformations and cleanup, and ensuring data consistency. Users, who can be in non-programming roles, can build visual pipelines to address business imperatives like regulatory compliance without having to wait for an IT team to write a Dataflow pipeline. Data Fusion also has a flexible API so IT staff can create scripts to automate execution.

Regardless of which ETL is used -- Dataflow, Dataproc, Data Fusion -- there are some crucial aspects to keep in mind. First: Maintaining data lineage is important. What do we mean by Lineage? Where the data came from, what processes it has been through, and what condition it is in, are all lineage. If you have the lineage, you know for what kinds of uses the data is suited. If you find the data gives odd results, you can check the lineage to find out if there is a cause that can be corrected. Lineage also helps with trust and regulatory compliance.

The other cross-cutting concern is that you need to keep metadata around. You need a way to track the lineage of data in your organization for discovery and identification of suitability for uses. On Google Cloud, Dataplex provides discoverability. But you have to do your bit by adding labels. Dataplex metadata can be viewed directly in BigQuery thereby simplifying the process of confirming data lineage.

A label is a key-value pair that helps you organize your resources. In BigQuery you can attach labels to Datasets, Tables, and Views. Labels are useful for managing complex resources because you can filter them based on their labels. Labels are a first step towards a Data Catalog. Among the things that labels help with is Cloud Billing. If you attach labels to Compute Engine instances and to buckets and to Dataflow pipelines, then you have a way to get a fine-grained look at your Cloud bill because the information about labels is forwarded to the billing system, and so you can break down your billing charges by label.

Data Catalog is a fully managed and highly scalable data discovery and metadata management service. It is serverless and requires no infrastructure to set up or manage. It provides access-level controls and honors source ACLs for read, write, and search for the data assets; giving you enterprise-grade access control. Think of Data Catalog as a metadata-as-a-service. It provides metadata management service for cataloging data assets via custom APIs and the UI, thereby providing a unified view of data wherever it is.

It supports schematized tags (e.g., Enum, Bool, DateTime) and not just simple text tags — providing organizations rich and organized business metadata. It offers unified data discovery of all data assets, spread across multiple projects and systems. It comes with a simple and easy-to-use search UI to quickly and easily find data assets; powered by the same Google search technology that supports Gmail and Drive.

As a central catalog, it provides a flexible and powerful cataloging system for capturing both technical metadata (automatically) as well as business metadata (tags) in a structured format. One of the great things about the data discovery is that it integrates with the Cloud Data Loss Prevention API. You can use it to discover and classify sensitive data, providing intelligence and helping to simplify the process of governing your data.

Data Catalog empowers users to annotate business metadata in a collaborative manner and provides the foundation for data governance. Specifically, Data Catalog makes all the metadata about your datasets available to search for your users, regardless of where the data are stored. Using Data Catalog, you can group datasets together with tags, flag certain columns as containing sensitive data, etc.

Why is this useful? If you have many different datasets with many different tables — to which different users have different access levels — the Data Catalog provides a single unified user experience for discovering those datasets quickly. No more hunting for specific table names in the databases, which may not be accessible by all users.

## The Hadoop Ecosystem

Let's start by looking at the Hadoop ecosystem in a little more detail. It helps to place the services you'll be learning about in historical context. Before 2006, big data meant big databases. Database design came from a time when storage was relatively cheap and processing was expensive, so it made sense to copy the data from its storage location to the processor to perform data processing, then the result would be copied back to storage.

Around 2006, distributed processing of big data became practical with Hadoop. The idea behind Hadoop is to create a cluster of computers and leverage distributed processing. HDFS, the Hadoop Distributed File System, stored the data on the machines in the cluster and MapReduce provided distributed processing of the data. A whole ecosystem of Hadoop related software grew up around Hadoop, including Hive, Pig and Spark.

Organizations use Hadoop for on-premises big data workloads. They make use of a range of applications that run on Hadoop clusters, such as Presto, but a lot of customers use Spark. Apache Hadoop is an open source software project that maintains the framework for distributed processing of large datasets across clusters of computers using simple programming models.

HDFS is the main file system Hadoop uses for distributing work to nodes on the cluster. Apache Spark is an open source software project that provides a high performance analytics engine for processing batch and streaming data. Spark can be up to 100 times faster than equivalent Hadoop jobs because it leverages in-memory processing. Spark also provides APIs for dealing with data including resilient distributed datasets and data frames. Spark in particular is very powerful and expressive and used for a lot of workloads.

A lot of the complexity and overhead of OSS Hadoop has to do with assumptions in the design that existed in the data center. Relieved of those limitations, data processing becomes a much richer solution with many more options. There are two common issues with OSS Hadoop: tuning and utilization. In many cases, using Dataproc as designed will overcome these limitations.

On-premises Hadoop clusters, due to their physical nature, suffer from limitations. The lack of separation between storage and compute resources results in capacity limits and an inability to scale fast. The only way to increase capacity is to add more physical servers.

There are many ways in which using Google Cloud can save you time, money and effort compared to using an on-premises Hadoop solution. In many cases, adopting a Cloud based approach can make your overall solution simpler and easier to manage.

Built-in support for Hadoop: Dataproc is a managed Hadoop and Spark environment. You can use Dataproc to run most of your existing jobs with minimal alteration, so you don't need to move away from all of the Hadoop tools you already know.

Managed hardware and configuration: When you run Hadoop on Google Cloud, you never need to worry about physical hardware. You specify the configuration of your cluster and Dataproc allocates resources for you, you can scale your cluster at any time.

Simplified version management: Keeping open source tools up to date and working together is one of the most complex parts of managing a Hadoop cluster. When you use Dataproc, much of that work is managed for you by Dataproc versioning.

Flexible job configuration: A typical on-premises Hadoop setup uses a single cluster that serves many purposes. When you move to Google Cloud, you can focus on individual tasks, creating as many clusters as you need. This removes much of the complexity of maintaining a single cluster with growing dependencies and software configuration interactions.

Running MapReduce directly on top of Hadoop is very useful, but it has the complication that the Hadoop system has to be tuned for the kind of job being run to make efficient use of the underlying resources. A simple explanation of Spark is that it is able to mix different kinds of applications and to adjust how it uses the available resources.

Spark uses a declarative programming model. In imperative programming, you tell the system what to do and how to do it. In declarative programming, you tell the system what you want and it figures out how to implement it. You will be learning to work with Spark in the labs in this course.

There is a full SQL implementation on top of Spark. There is a common data frame model that works across Scala, Java, Python, SQL and R. And there is a distributed machine learning library called Spark ML Lib.

## Running Hadoop on Dataproc

Next, we'll discuss how and why you should consider processing your same Hadoop job code in the cloud using Dataproc on Google Cloud. Dataproc lets you take advantage of open source data tools for batch processing, querying, streaming, and machine learning. Dataproc automation helps you create clusters quickly, manage them easily, and save money by turning clusters off when you don't need them.

When compared to traditional, on-premises products, and competing cloud services, Dataproc has unique advantages for clusters of three to hundreds of nodes. There is no need to learn new tools or APIs to use Dataproc, making it easy to move existing projects into Dataproc without redevelopment. Spark, Hadoop, Pig, and Hive are frequently updated.

Here are some of the key features of Dataproc:

Low cost: Dataproc is priced at 1 cent per virtual CPU per cluster per hour, on top of the other Google Cloud resources you use. In addition, Dataproc clusters can include preemptible instances that have lower compute prices. You use and pay for things only when you need them, so Dataproc charges second-by-second billing with a one-minute-minimum billing period.

Super-fast: Dataproc clusters are quick to start, scale, and shutdown, with each of these operations taking 90 seconds or less, on average. Resizable clusters: Clusters can be created and scaled quickly with a variety of virtual machine types, disk sizes, number of nodes, and networking options.

Open source ecosystem: You can use Spark and Hadoop tools, libraries, and documentation with Dataproc. Dataproc provides frequent updates to native versions of Spark, Hadoop, Pig, and Hive, so there is no need to learn new tools or APIs, and it is possible to move existing projects or ETL pipelines without redevelopment.

Integrated: Built-in integration with Cloud Storage, BigQuery, and Bigtable ensures data will not be lost. This, together with Cloud Logging and Cloud Monitoring, provides a complete data platform and not just a Spark or Hadoop cluster. For example, you can use Dataproc to effortlessly ETL terabytes of raw log data directly into BigQuery for business reporting.

Managed: Easily interact with clusters and Spark or Hadoop jobs, without the assistance of an administrator or special software, through the Cloud Console, the Cloud SDK, or the Dataproc REST API. When you're done with a cluster, simply turn it off, so money isn't spent on an idle cluster.

Versioning: Image versioning allows you to switch between different versions of Apache Spark, Apache Hadoop, and other tools. Highly available: Run clusters with multiple primary nodes and set jobs to restart on failure to ensure your clusters and jobs are highly available.

Developer tools: Multiple ways to manage a cluster, including the Cloud Console, the Cloud SDK, RESTful APIs, and SSH access. Initialization actions: Run initialization actions to install or customize the settings and libraries you need when your cluster is created. And automatic or manual configuration: Dataproc automatically configures hardware and software on clusters for you while also allowing for manual control.

Dataproc has two ways to customize clusters: optional components and initialization actions. Pre-configured optional components can be selected when deploying from the console or via the command line and include: Anaconda, Hive WebHCat, Jupyter Notebook, Zeppelin Notebook, Druid, Presto and Zookeeper.

Initialization actions let you customize your cluster by specifying executables or scripts that Dataproc will run on all nodes in your Dataproc cluster immediately after the cluster is set up. Here's an example of how you can create a Dataproc cluster using the Cloud SDK, and we're going to specify an HBase shell script to run on the clusters initialization. There are a lot of pre-built startup scripts that you can leverage for common Hadoop cluster setup tasks, like Flink, Jupyter and more. You can check out the GitHub repo link in the Course Resources to learn more.

Let's talk more about the architecture of the cluster. A Dataproc cluster can contain either preemptible secondary workers or non-preemptible secondary workers, but not both. The standard setup architecture is much like you would expect on-premise. You have a cluster of virtual machines for processing and then persistent disks for storage via HDFS.

You've also got your manager node VM (or VMs) and a set of worker nodes. Worker nodes can also be part of a managed instance group, which is just another way of ensuring that VMs within that group are all of the same template. The advantage is that you can spin up more VMs than you need to automatically resize your cluster based on the demands.

Google Cloud recommends a ratio of 60/40 as the maximum between standard VMs and preemptible VMs. Generally, you shouldn't think of a Dataproc cluster as long-lived. Instead you should spin them up when you need compute processing for a job and then simply turn them down. You can also persist them indefinitely if you want to.

What happens to HDFS storage on disk when you turn those clusters down? The storage would go away too, which is why it's a best practice to use storage that's off cluster by connecting to other Google Cloud products. Instead of using native HDFS on a cluster, you could simply use a cluster on Cloud Storage via the HDFS connector. It's pretty easy to adapt existing Hadoop code to use Cloud Storage instead of HDFS. Change the prefix for this storage from hdfs// to gs//. What about Hbase off-cluster? Consider writing to Bigtable instead. What about large analytical workloads? Consider writing that data into BigQuery and doing those analytical work loads there.

Using Dataproc involves this sequence of events: Setup, Configuration, Optimization, Utilization, and Monitoring. Setup means creating a cluster, and you can do that through the Cloud Console, or from the command line using the gcloud command. You can also export a YAML file from an existing cluster or create a cluster from a YAML file. You can create a cluster from a Terraform configuration, or use the REST API.

The cluster can be set as a single VM, which is usually to keep costs down for development and experimentation. Standard is with a single Primary Node, and High Availability has three Primary Nodes. You can choose a region and zone, or select a "global region" and allow the service to choose the zone for you. The cluster defaults to a Global endpoint, but defining a Regional endpoint may offer increased isolation and in certain cases, lower latency.

The Primary Node is where the HDFS Namenode runs, as well as the YARN node and job drivers. HDFS replication defaults to 2 in Dataproc. Optional components from the Hadoop-ecosystem include: Anaconda (Python distribution and package manager), Hive Webcat, Jupyter Notebook, and Zeppelin Notebook.

Cluster properties are run-time values that can be used by configuration files for more dynamic startup options. User labels can be used to tag the cluster for your own solutions or reporting purposes. The Primary Node Worker Nodes, and preemptible Worker Nodes, if enabled, have separate VM options, such as vCPU, memory, and storage.

Preemptible nodes include YARN NodeManager but they do not run HDFS. There are a minimum number of worker nodes, the default is 2. The maximum number of worker nodes is determined by a quota and the number of SSDs attached to each worker. You can also specify initialization actions, such as initialization scripts that can further customize the worker nodes. Metadata can be defined so that the VMs can share state information.

Preemptible VMs can be used to lower costs. Just remember they can be pulled from service at any time and within 24 hours. So your application might need to be designed for resilience to prevent data loss. Custom machine types allow you to specify the balance of Memory and CPU to tune the VM to the load, so you are not wasting resources.

A custom image can be used to pre-install software so that it takes less time for the customized node to become operational than if you installed the software at boot-time using an initialization script. You can also use a Persistent SSD boot disk for faster cluster start-up.

Jobs can be submitted through the cloud console, the gcloud command, or the REST API. They can also be started by orchestration services such as Dataproc Workflow and Cloud Composer. Don't use Hadoop's direct interfaces to submit jobs because the metadata will not be available to Dataproc for job and cluster management, and for security, they are disabled by default.

By default, jobs are not restartable. However, you can create restartable jobs through the command line or REST API. Restartable jobs must be designed to be idempotent and to detect successorship and restore state.

Lastly, after you submit your job you'll want to monitor it, you can do so using Cloud Monitoring. Or you can also build a custom dashboard with graphs and set up monitoring of alert policies to send emails for example, where you can notify if incidents happen. Any details from HDFS, YARN, metrics about a particular job or overall metrics for the cluster like CPU utilization, disk and network usage, can all be monitored and alerted on with Cloud Monitoring.

## Cloud Storage instead of HDFS

Let's discuss more about using Google Cloud Storage instead of the native Hadoop file system or HDFS. Network speeds were slow originally. That's why we kept data as close as possible to the processor. Now, with petabit networking, you can treat storage and compute independently and move traffic quickly over the network.

Your on-premise Hadoop clusters need local storage on its disk since the same server runs computes on stores jobs. That's one of the first areas for optimization. You can run HDFS in the Cloud just by lifting and shifting your Hadoop workloads to Dataproc. This is often the first step to the Cloud and requires no code changes. It just works.

But HDFS on the Cloud is a subpar solution in the long run. This is because of how HDFS works on the clusters with block size, the data locality and the replication of the data in HDFS. For block size in HDFS, you're tying the performance of input and output to the actual hardware the server is running on.

Again, storage is not elastic in this scenario, you're in the cluster. If you run out of persistent disk space on your cluster, you will need to resize even if you don't need the extra compute power. For data locality, there are similar concerns about storing data on individual persistent disks. This is especially true when it comes to replication. In order for HDFS to be highly available, it replicates three copies of each block out to storage.

It would be better to have a storage solution that's separately managed from the constraints of your cluster. Google's network enables new solutions for big data. The Jupyter networking fabric within a Google data center delivers over one petabit per second of bandwidth. To put that into perspective, that's about twice the amount of traffic exchanged on the entire public internet. See Cisco's annual estimate of all internet traffic.

If you draw a line somewhere in a network, bisectional bandwidth is the rate of communication at which servers on one side of the line can communicate with servers on the other side. With enough bisectional bandwidth, any server can communicate with any other server at full network speeds. With petabit bisectional bandwidth, the communication is so fast that it no longer makes sense to transfer files and store them locally. Instead, it makes sense to use the data from where it is stored.

Inside of a Google data center, the internal name for the massively distributed storage layer is called Colossus. Under the network inside the data center is Jupyter. Dataproc clusters get the advantage of scaling up and down VMs that they need to do the compute while passing off persistent storage needs with the ultra-fast Jupyter network to a storage products like Cloud Storage, which is controlled by Colossus behind the scenes.

A historical continuum of data management is as follows. Before 2006, big data meant big databases, database design came from a time when storage was relatively cheap, and processing was expensive. Around 2006, distributed processing of big data became practical with Hadoop. Around 2010, BigQuery was released, which was the first of many big data services developed by Google. Around 2015, Google launched Dataproc, which provides a managed service for creating Hadoop and Spark clusters and managing data processing workloads.

One of the biggest benefits of Hadoop in the Cloud is that separation of compute and storage. With Cloud Storage as the backend, you can treat clusters themselves as ephemeral resources, which allows you not to pay for compute capacity when you're not running any jobs. Also, Cloud Storage is its own completely scalable and durable storage service, which is connected to many other Google Cloud projects.

Cloud storage could be a drop-in replacement for your HDFS backend for Hadoop, the rest of your code would just work. Also, you can use the Cloud storage connector manually on your non-Cloud Hadoop clusters if you didn't want to migrate your entire cluster to the Cloud yet.

With HDFS, you must over-provision for current data and for data you might have, and you must use persistent disks throughout. With Cloud Storage however, you pay for exactly what you need when you use it. Cloud Storage is optimized for large bulk parallel operations. It has very high throughput, but it has significant latency. If you have large jobs that are running lots of tiny little blocks, you may be better off with HDFS. Additionally, you want to avoid iterating sequentially over many nested directories in a single job.

Using Cloud Storage instead of HDFS provides some key benefits due to the distributed service including eliminating bottlenecks and single points of failure. However, there are some disadvantages to be aware of, including the challenges presented by renaming objects and the inability to append to objects. Cloud Storage is at its core an object store, it only simulates a file directory. So directory renames in HDFS are not the same as they are in Cloud Storage, but new objects store oriented output committers mitigate this as you see here.

Disk CP is a key tool for moving data. In general, you want to use a push-based model for any data that you know you will need while pull-based may be a useful model if there is a lot of data that you might not ever need to migrate.

## Optimizing Dataproc

Next, let's look at optimizing Dataproc. Where is your data and where is your cluster? Knowing your data locality can have a major impact on your performance. You want to be sure that your data's region and your cluster zone are physically close in distance.

When using Dataproc, you can omit the zone and have the Dataproc Auto Zone feature select a zone for you in the region you choose. While this handy feature can optimize where to put your cluster, it does not know how to anticipate the location of the data your cluster will be accessing. Make sure that the Cloud storage bucket is in the same regional location as your Dataproc region.

Is your network traffic being funneled? Be sure that you do not have any network rules or routes that funnel Cloud storage traffic through a small number of VPN gateways before it reaches your cluster. There are large network pipes between Cloud Storage and Compute Engine. You don't want to throttle your bandwidth by sending traffic into a bottleneck in your Google Cloud networking configuration.

How many input files and Hadoop partitions are you trying to deal with? Make sure you are not dealing with more than around 10,000 input files. If you find yourself in this situation, try to combine or union the data into larger file sizes. If this file volume reduction means that now you are working with larger datasets more than approximately 50,000 Hadoop partitions, you should consider adjusting the setting fs.gs.block.size to a larger value accordingly.

Is the size of your persistent disk limiting your throughput? Oftentimes when getting started with Google Cloud, you may have just a small table that you want to benchmark. This is generally a good approach as long as you do not choose a persistent disk that assigns to such a small quantity of data, it will most likely limit your performance. Standard persistent disk scales linearly with volume size.

Did you allocate enough virtual machines to your cluster? A question that often comes up when migrating from on-premises hardware to Google Cloud is how to accurately size the number of virtual machines needed. Understanding your workloads is key to identifying a cluster size. Running prototypes and benchmarking with real data and real jobs is crucial to informing the actual VM allocation decision.

Locally, the ephemeral nature of the Cloud makes it easy to write size clusters for the specific task at hand instead of trying to purchase hardware upfront, thus, you can easily resize your cluster as needed. Employing job scoped clusters is a common strategy for Dataproc clusters.

## Optimizing Dataproc Storage

Local HDFS is a good option if your jobs require a lot of metadata operations—for example, you have thousands of partitions and directories, and each file size is relatively small. You modify the HDFS data frequently or you rename directories. Cloud Storage objects are immutable, so renaming a directory is an expensive operation because it consists of copying all objects to a new key and deleting them afterwards. You heavily use the append operation on HDFS files. You have workloads that involve heavy I/O. For example, you have a lot of partitioned writes, such as in this example. You have I/O workloads that are especially sensitive to latency. For example, you require single-digit millisecond latency per storage operation.

In general, we recommend using Cloud Storage as the initial and final source of data in a big-data pipeline. For example, if a workflow contains five Spark jobs in series, the first job retrieves the initial data from Cloud Storage and then writes shuffle data and intermediate job output to HDFS. The final Spark job writes its results to Cloud Storage.

Using Dataproc with Cloud Storage allows you to reduce the disk requirements and save costs by putting your data there instead of in the HDFS. When you keep your data on Cloud Storage and don't store it on the local HDFS, you can use smaller disks for your cluster. By making your cluster truly on-demand, you're also able to separate storage and compute, as noted earlier, which helps you reduce costs significantly.

Even if you store all of your data in Cloud Storage, your Dataproc cluster needs HDFS for certain operations such as storing control and recovery files, or aggregating logs. It also needs non-HDFS local disk space for shuffling. You can reduce the disk size per worker if you are not heavily using the local HDFS.

Here are some options to adjust the size of the local HDFS. Decrease the total size of the local HDFS by decreasing the size of primary persistent disks for the primary and workers. The primary persistent disk also contains the boot volume and system libraries, so allocate at least 100 GB. Increase the total size of the local HDFS by increasing the size of primary persistent disk for workers. Consider this option carefully— it's rare to have workloads that get better performance by using HDFS with standard persistent disks in comparison to using Cloud Storage or local HDFS with SSD.

Attach up to eight SSDs to each worker and use these disks for the HDFS. This is a good option if you need to use the HDFS for I/O-intensive workloads and you need single-digit millisecond latency. Make sure that you use a machine type that has enough CPUs and memory on the worker to support these disks. And use SSD persistent disks for your primary or workers as a primary disk.

You should understand the repercussions of geography and regions before you configure your data and jobs. Many Google Cloud services require you to specify regions or zones in which to allocate resources. The latency of requests can increase when the requests are made from a different region than the one where the resources are stored. Additionally, if the service's resources and your persistent data are located in different regions, some calls to Google Cloud services might copy all of the required data from one zone to another before processing. This can have a severe impact on performance.

Cloud Storage is the primary way to store unstructured data in Google Cloud, but it isn't the only storage option. Some of your data might be better suited to storage in products designed explicitly for big data. You can use Bigtable to store large amounts of sparse data. Bigtable is an HBase-compliant API that offers low latency and high scalability to adapt to your jobs. For data warehousing, you can use BigQuery.

Because Dataproc runs Hadoop on Google Cloud, using a persistent Dataproc cluster to replicate your on-premises setup might seem like the easiest solution. However, there are some limitations to that approach. Keeping your data in a persistent HDFS cluster using Dataproc is more expensive than storing your data in Cloud Storage, which is what we recommend. Keeping data in an HDFS cluster also limits your ability to use your data with other Google Cloud products.

Augmenting or replacing some of your open-source-based tools with other related Google Cloud services can be more efficient or economical for particular use cases. Using a single, persistent Dataproc cluster for your jobs is more difficult to manage than shifting to targeted clusters that serve individual jobs or job areas.

The most cost-effective and flexible way to migrate your Hadoop system to Google Cloud is to shift away from thinking in terms of large, multi-purpose, persistent clusters and instead think about small, short-lived clusters that are designed to run specific jobs. You store your data in Cloud Storage to support multiple, temporary processing clusters. This model is often called the ephemeral model, because the clusters you use for processing jobs are allocated as needed and are released as jobs finish.

If you have efficient utilization, don't pay for resources that you don't use - employ scheduled deletion. A fixed amount of time after the cluster enters an idle state, you can automatically set a timer. You can give it a timestamp, and the count starts immediately once the expiration has been set. You can set a duration, the time in seconds to wait before automatically turning down the cluster. You can range from ten minutes as a minimum, to 14 days as a maximum at a granularity of one second.

The biggest shift in your approach between running an on-premises Hadoop workflow and running the same workflow on Google Cloud is the shift away from monolithic, persistent clusters to specialized, ephemeral clusters. You spin up a cluster when you need to run a job and then delete it when the job completes. The resources required by your jobs are active only when they're being used, so you only pay for what you use.

To get the most from Dataproc, customers need to move to an "ephemeral" model of only using clusters when they need them. This can be scary because a persistent cluster is comfortable. With Cloud Storage data persistence and fast boot of Dataproc, however, a persistent cluster is a waste of resources. If a persistent cluster is needed, make it small. Clusters can be resized anytime.

Ephemeral model is the recommended route but it requires storage to be decoupled from compute. Separate job shapes and separate clusters. Decompose even further with job-scoped clusters. Isolate dev, staging, and production environments by running on separate clusters. Read from the same underlying data source on Cloud Storage. Add appropriate ACLs to service accounts to protect data.

The point of ephemeral clusters is to use them only for the jobs' lifetime. When it's time to run a job, follow this process: Create a properly configured cluster. Run your job, sending output to Cloud Storage or another persistent location. Delete the cluster. Use your job output however you need to. View logs in Cloud Logging or Cloud Storage.

If you can't accomplish your work without a persistent cluster, you can create one. This option may be costly and isn't recommended if there is a way to get your job done on ephemeral clusters. You can minimize the cost of a persistent cluster by: Creating the smallest cluster you can, scoping your work on that cluster to the smallest possible number of jobs, and scaling the cluster to the minimum workable number of nodes, adding more dynamically to meet demand.


## Optimizing Dataproc templates and autoscaling

The Dataproc Workflow Template is a YAML file that is processed through a Directed Acyclic Graph or DAG. It can create a new cluster, select from an existing cluster, submit jobs, hold jobs for submission until dependencies can complete, and it can delete a cluster when the job is done. It's available through the gcloud command and the REST API.

You can view existing workflow templates and instantiated workflows through the Google Cloud console as well. The Workflow Template becomes active when it is instantiated into the DAG. The Template can be submitted multiple times with different parameter values. You can also write a template inline in a gcloud command, and you can list workflows and workflow metadata to help diagnose issues.

Here's an example of a Dataproc workflow template. First, we get all the things that need to be installed in the cluster using our startup scripts and manually echoing pip install commands like the one seen here to install matplotlib. You can have multiple startup shell scripts run like you see in this example.

Next, we use the gcloud command for creating a new cluster in advance of running our job. We specify cluster parameters like the template to be used in our desired architecture and what machine types and image versions we want for hardware and software. After that, we need to add a job to the newly created cluster. In this example, we have a Spark job written in Python that exists in a Cloud Storage bucket that we control. Lastly, we need to submit this template itself as a new workflow template as you see with the last command.

Dataproc autoscaling provides clusters that size themselves to the needs of the enterprise. Key features include: Jobs are "fire and forget", there's no need to manually intervene when a cluster is over or under capacity, you can choose between standard and preemptible workers, and you can save resources such as quota and cost at any point in time.

Autoscaling policies provide fine-grained control. This is based on the difference between YARN pending and available memory. If more memory is needed, then you scale up. If there's excess memory, you scale down. Obey VM limits and scale based on scale factor.

Dataproc autoscaling provides flexible capacity for more efficient utilization, making scaling decisions based on Hadoop YARN Metrics. It's designed to be used only with off-cluster persistent data, not on-cluster HDFS or HBase. It works best with a cluster that processes a lot of jobs or that processes a single large job. It doesn't support Spark Structured Streaming, a streaming service built on top of Spark SQL. It's also not designed to scale to zero, so it's not the best for sparsely utilized or idle clusters. In these cases it's equally fast to terminate a cluster that's idle and create a new cluster when it's needed. For that purpose you would look at Dataproc Workflows or Cloud Composer, and Cluster Scheduled Deletion.

One of the things that you want to consider when working with autoscaling is setting the initial workers. The number of initial workers is set from Worker Nodes, Nodes Minimum. Setting this value ensures that the cluster comes up to basic capacity faster than if you let autoscaling handle it, because autoscaling might require multiple autoscale periods to scale up.

The primary minimum number of workers may be the same as the cluster nodes minimum. There is a maximum that caps the number of worker nodes. If there is heavy load on the cluster, autoscaling determines it is time to scale up. The scale_up.factor determines how many nodes to launch. This would commonly be one node, but if you knew that a lot of demand would occur at once, maybe you want to scale up faster.

After the action, there is a cooldown period to let things settle before autoscaling evaluation occurs again. The cooldown period reduces the chances that the cluster will start and terminate nodes at the same time. In this example, the extra capacity isn't needed, and there is a graceful decommission timeout to give running jobs a chance to complete before the node goes out of service.

Notice there is a scale down factor. In this case it is scaling down by one node at a time for a more leisurely reduction in capacity. After the action, there is another cooldown period and a second scale down, resulting in a return to the minimum number of workers. A secondary min_workers and max_workers controls the scale of preemptible workers.

## Optimizing Dataproc monitoring

In Google Cloud, you can use Cloud Logging and Cloud Monitoring to view and customize logs, and to monitor jobs and resources.

The best way to find what error caused a Spark job failure is to look at the driver output and the logs generated by the Spark executioners.

Note, however, that if you submit a Spark job by connecting directly to the primary node using SSH, it's not possible to get the driver output.

You can retrieve the driver program output by using the Cloud Console or by using G Cloud command. The output is also stored in the Cloud storage bucket of the Dataproc cluster.

All other logs are located in different files inside the machines of the cluster. It's possible to see the logs for each container from the spark app Web UI, or from the history server after the program ends in the executer's tab. You need to browse through each Spark container to view each log.

If you write logs or print to standard out or standard air in your application code, the logs are saved in the redirection of standard out or standard air. In a Dataproc cluster, Yarn is configured to collect all these logs by default, and they're available in Cloud Logging.

Logging provides a consolidated and concise view of all logs so that you don't need to spend time browsing among container logs to find errors. This screen shows the login page in the Cloud Console. You can view all logs from your Dataproc cluster by selecting the cluster's name in the selector menu. Don't forget to expand the time duration in the time range selector.

You can get logs from a Spark application by filtering by its ID, you can get the application ID from the driver output. To find logs faster, you can create and use your own labels for each cluster or for each Dataproc job.

For example, you can create a label with the key environment or ENV as the value in the exploration and use it for your data exploration job. You can then get logs for all exploration job creations by filtering with the label environment with a value exploration in logging. Note that this filter will not return all logs for this job, only the resource creation logs.

You can set the driver log level using the following G Cloud command: G Cloud, Dataproc, jobs, submit, Hadoop, with the parameter driver log levels. You set the log level for the rest of the application from the spark context, for example, Spark dot Spark context dot set log level. And for here, we'll just say the example is debug.

Cloud Monitoring can monitor the cluster's CPU, disk, network usage and Yarn resources. You can create a custom dashboard to get up to date charts for these and other metrics. Dataproc runs on top of Compute Engine. If you want to visualize CPU usage, disk IO or networking metrics in a chart, you need to select a Compute Engine VM instance as the resource type, and then filter by the cluster name.

This diagram shows an example of the output. To view metrics for Spark queries, jobs, stages, or tasks, connect to the spark applications Web UI.
