# Week 2: GCCF: Infrastructure in Google Cloud Part 1

## Storage Options in the Cloud

Let's begin by exploring the different storage options available in Google Cloud. Every application needs to store data, like media to be streamed or perhaps even sensor data from devices, and different applications and workloads require different storage database solutions.

Google Cloud offers relational and non-relational databases and worldwide object storage. These options help users save costs, reduce the time it takes to launch, and make the most of their datasets by being able to analyze a wide variety of data.

Choosing the right option to store and process data often depends on the data type that needs to be stored and the business need. Google Cloud provides managed storage and database services that are scalable, reliable, and easy to operate.

These include Cloud Storage, Cloud SQL, Cloud Spanner, Firestore, and Cloud Bigtable. The goal of these products is to reduce the time and effort needed to store data. This means creating an elastic storage bucket directly in a web interface or through a command line.

There are three common cloud storage use cases. The first is content storage and delivery. This is when content, such as images or videos, needs to be served to users wherever they are. People want their content fast, so running on the global network that Google provides makes for a great experience for end users.

The second use case is storage for data analytics and general compute. Users can process or expose their data to analytics tools, like the analytics stack of products that Google Cloud offers, and do things like genomic sequencing or IoT data analysis.

The third use case is backup and archival storage. Users can save storage costs by migrating infrequently accessed content to cheaper cloud storage options. Also, if anything happens to their data on-premises, it's critical to have a copy in the cloud for recovery purposes.

For users with databases, Google's first priority is to help them migrate existing databases to the cloud and move them to the right service. This will usually be users moving MySQL or Postgre workloads to Cloud SQL. The second priority is to help users innovate, build or rebuild for the cloud, offer mobile applications, and plan for future growth.

## Structured and Unstructured Data Storage

Now you'll learn to differentiate between structured and unstructured storage in the cloud. Let's start with unstructured versus structured data.

Unstructured data is information stored in a non-tabular form such as documents, images, and audio files. Unstructured data is usually best suited to Cloud Storage. It's estimated that around 80 percent of all data is unstructured. It's far more difficult to process or analyze unstructured data using traditional methods because the data has no internal identifier to enable search functions to identify it. Unstructured data often includes text and multimedia content, for example, email messages, documents, photos, videos, presentations, and web pages. Organizations are focusing increasingly on mining unstructured data for insights that will provide them with a competitive edge.

Alternatively, there is structured data, which represents information stored in tables, rows, and columns. Structured data is what most people are used to working with and typically fits within columns and rows in spreadsheets or relational databases. You can expect this type of data to be organized and clearly defined and usually easy to capture, access, and analyze. Examples of structured data include names, addresses, contact numbers, dates, and billing info. The benefit of structured data is that it can be understood by programming languages and can be manipulated relatively quickly.

Structured data comes in two types: transactional workloads and analytical workloads. Transactional workloads stem from Online Transaction Processing systems, which are used when fast data inserts and updates are required to build row-based records. This is usually to maintain a system snapshot. They require relatively standardized queries that affect only a few records.

If your data is transactional and you need to access it using SQL, Cloud SQL and Cloud Spanner are two options. Cloud SQL works best for local to regional scalability, but Cloud Spanner is best to scale a database globally. If the transactional data will be accessed without SQL, Firestore might be the best option. Firestore is a transactional NoSQL, document-oriented database.

Then there are analytical workloads, which stem from Online Analytical Processing systems, which are used when entire datasets need to be read. They often require complex queries, for example, aggregations. If you have analytical workloads that require SQL commands, BigQuery may be the best option. BigQuery, Google's data warehouse solution, lets you analyze petabyte-scale datasets. Alternatively, Bigtable provides a scalable NoSQL solution for analytical workloads. It's best for real-time, high-throughput applications that require only millisecond latency.

## Unstructured Data Storage Using Cloud Storage

Now you'll learn to differentiate between structured and unstructured storage in the cloud. Let's start with unstructured versus structured data.

Unstructured data is information stored in a non-tabular form such as documents, images, and audio files. Unstructured data is usually best suited to Cloud Storage. It's estimated that around 80 percent of all data is unstructured. It's far more difficult to process or analyze unstructured data using traditional methods because the data has no internal identifier to enable search functions to identify it. Unstructured data often includes text and multimedia content, for example, email messages, documents, photos, videos, presentations, and web pages. Organizations are focusing increasingly on mining unstructured data for insights that will provide them with a competitive edge.

Alternatively, there is structured data, which represents information stored in tables, rows, and columns. Structured data is what most people are used to working with and typically fits within columns and rows in spreadsheets or relational databases. You can expect this type of data to be organized and clearly defined and usually easy to capture, access, and analyze. Examples of structured data include names, addresses, contact numbers, dates, and billing info. The benefit of structured data is that it can be understood by programming languages and can be manipulated relatively quickly.

Structured data comes in two types: transactional workloads and analytical workloads. Transactional workloads stem from Online Transaction Processing systems, which are used when fast data inserts and updates are required to build row-based records. This is usually to maintain a system snapshot. They require relatively standardized queries that affect only a few records.

So, if your data is transactional and you need to access it using SQL, Cloud SQL and Cloud Spanner are two options. Cloud SQL works best for local to regional scalability, but Cloud Spanner is best to scale a database globally. If the transactional data will be accessed without SQL, Firestore might be the best option. Firestore is a transactional NoSQL, document-oriented database.

Then there are analytical workloads, which stem from Online Analytical Processing systems, which are used when entire datasets need to be read. They often require complex queries, for example, aggregations. If you have analytical workloads that require SQL commands, BigQuery may be the best option. BigQuery, Google's data warehouse solution, lets you analyze petabyte-scale datasets. Alternatively, Bigtable provides a scalable NoSQL solution for analytical workloads. It's best for real-time, high-throughput applications that require only millisecond latency.

## SQL Managed Services

Now we'll explore the use case for SQL-managed services. Let's revisit what a database is and how it's used. A database is a collection of information that is organized so that it can easily be accessed and managed.

Users are building software applications using databases to answer business questions such as buying a ticket, filing an expense report, storing a photo, or storing medical records. Computer applications run databases to get a fast answer to questions like: What's this user's name, given their sign-in information, so I can display it? What's the cost of product Y so I can show it on my dynamic web page? What were my top 10 best selling products this month? What is the next ad that I should show the user currently browsing my site?

These apps must be able to write data in and read data out of databases. When a database is used, it's usually run by a computer application. So when we say that "a database is useful for doing X," it's usually because it's designed to make answering a question simple, fast, and efficient for the app.

Relational database management systems, abbreviated RDBMS, or just relational databases, are used extensively and are the kind of database you encounter most of the time. They're organized based on the relational model of data. They are very good when you have a well-structured data model and when you need transactions and the ability to join data across tables to retrieve complex combinations of your data.

Because they make use of the Structured Query Language, they are sometimes called SQL databases. Google Cloud offers two managed relational database services, Cloud SQL and Cloud Spanner. Let's explore each of them in detail.

## Exploring Cloud SQL

We'll begin with Cloud SQL. Cloud SQL offers fully managed relational databases, including MySQL, PostgreSQL, and SQL Server as a service. It's designed to hand off mundane, but necessary and often time-consuming, tasks to Google—like applying patches and updates, managing backups, and configuring replications—so your focus can be on building great applications.

Cloud SQL doesn't require any software installation or maintenance. It can scale up to 96 processor cores, 624 GB of RAM, and 64 TB of storage. Cloud SQL supports automatic replication scenarios, such as from a Cloud SQL primary instance, an external primary instance, and external MySQL instances.

The service supports managed backups, so backed-up data is securely stored and accessible if a restore is required. The cost of an instance covers seven backups. Cloud SQL encrypts customer data when on Google's internal networks and when stored in database tables, temporary files, and backups. It includes a network firewall, which controls network access to each database instance.

## Cloud Spanner as a Managed Service

In this next section, you'll explore how Cloud Spanner can be leveraged as a managed service. Cloud Spanner is a fully managed relational database service that scales horizontally, is strongly consistent, and speaks SQL.

Vertical scaling is where you make a single instance larger or smaller, while horizontal scaling is when you scale by adding and removing servers. Cloud Spanner users are often in the advertising, finance, and marketing technology industries, where they need to manage end-user metadata.

Tested by Google's own mission-critical applications and services, Spanner is the service that powers Google's $80 billion business. Cloud Spanner is especially suited for applications that require an SQL relational database management system with joins and secondary indexes, built-in high availability, strong global consistency, and high numbers of input/output operations per second, such as tens of thousands of reads/writes per second.

So how does Cloud Spanner work? Data is automatically and instantly copied across regions, which is called synchronous replication. As a result, queries always return consistent and ordered answers regardless of the region. Google uses replication within and across regions to achieve availability, so if one region goes offline, a user's data can still be served from another region.

## NoSQL Managed Services Options

Now we'll explore the NoSQL managed services options currently available.

Google offers two managed NoSQL database options, Firestore and Cloud Bigtable.

Firestore is a fully managed, serverless NoSQL document store that supports ACID transactions.

Cloud Bigtable is a petabyte scale, sparse wide column NoSQL database that offers extremely low write latency.

Let's explore each option in detail.

## Firestore, a NoSQL Document Store

In this next section, you'll explore Firestore, a database that lets you develop rich applications using a fully managed, scalable, and serverless document database.

Firestore is a flexible, horizontally scalable, NoSQL cloud database for mobile, web, and server development. With Firestore, incoming data is stored in a document structure, and these documents are then organized into collections. Documents can contain complex nested objects in addition to subcollections.

Firestore's NoSQL queries can then be used to retrieve individual, specific documents or to retrieve all the documents in a collection that match your query parameters. Queries can include multiple, chained filters and combine filtering and sorting options. They're also indexed by default, so query performance is proportional to the size of the result set, not the dataset.

Firestore uses data synchronization to update data on any connected device. However, it's also designed to make simple, one-time fetch queries efficiently. It caches data that an app is actively using, so the app can write, read, listen to, and query data even if the device is offline. When the device comes back online, Firestore synchronizes any local changes back to Firestore.

Firestore leverages Google Cloud's powerful infrastructure: automatic multi-region data replication, strong consistency guarantees, atomic batch operations, and real transaction support.

## Bigtable as a NoSQL option

The final topic of this module describes how to leverage Cloud Bigtable as a NoSQL option. Bigtable is Google's NoSQL big data database service. It's the same database that powers many core Google services, including Search, Analytics, Maps, and Gmail.

Bigtable is designed to handle massive workloads at consistent low latency and high throughput, so it's a great choice for both operational and analytical applications, including Internet of Things, user analytics, and financial data analysis.

When deciding which storage option is best, customers often choose Bigtable if they're working with more than 1TB of semi-structured or structured data. Data is fast with high throughput, or it's rapidly changing. They're working with NoSQL data. This usually means transactions where strong relational semantics are not required. Data is a time-series or has natural semantic ordering. They're working with big data, running asynchronous batch or synchronous real-time processing on the data. Or they're running machine learning algorithms on the data.

Bigtable can interact with other Google Cloud services and third-party clients. Using APIs, data can be read from and written to Bigtable through a data service layer like Managed VMs, the HBase REST Server, or a Java Server using the HBase client. Typically this is used to serve data to applications, dashboards, and data services.

Data can also be streamed in through various popular stream processing frameworks like Dataflow Streaming, Spark Streaming, and Storm. And if streaming is not an option, data can also be read from and written to Bigtable through batch processes like Hadoop MapReduce, Dataflow, or Spark. Often, summarized or newly calculated data is written back to Bigtable or to a downstream database.

BigQuery is not mentioned in this module because it sits on the edge between data storage and data processing and is covered in more depth in other courses. The usual reason to store data in BigQuery is so you can use its big data analysis and interactive querying capabilities. It is not purely a data storage product.

## Apigee Edge

Another Google Cloud platform available for developing and managing API proxies is Apigee API Management. Unlike Cloud Endpoints, Apigee API Management has a specific focus on business problems, like rate limiting, quotas, and analytics. In fact, many Apigee API Management users provide a software service to other companies.

Backend services for Apigee API Management don't have to be in Google Cloud, and as a result, engineers also often use it to take apart legacy applications. So, instead of replacing a large, important application in one move, they can use Apigee API Management to peel off its services individually. This allows them to stand up microservices to implement each in turn, until the legacy application can finally be retired.

## Pub/Sub

Pub/Sub is a Google Cloud asynchronous messaging service and API that supports distributed message-oriented architectures at scale. One of the early stages in a data pipeline is data ingestion, which is where large amounts of streaming data are received. Data, however, may not always come from a single, structured database. Instead, the data might stream from a thousand, or even a million, different events that are all happening asynchronously.

A common example of this is data from IoT, or Internet of Things, applications. These can include sensors on taxis that send out location data every 30 seconds or temperature sensors around a data center to help optimize heating and cooling.

These IoT devices present new challenges to data ingestion, which can be summarized as follows: The first is that data can be streamed from many different methods and devices, many of which might not talk to each other and might be sending bad or delayed data. The second is that it can be hard to distribute event messages to the right subscribers. Event messages are notifications. A method is needed to collect the streaming messages that come from IoT sensors and broadcast them to the subscribers as needed. The third is that data can arrive quickly and at high volumes. Services must be able to support this. And the fourth challenge is ensuring that services are reliable and secure, and perform as expected.

The name Pub/Sub is short for Publisher/Subscriber, or publish messages to subscribers. Pub/Sub is a distributed messaging service that can receive messages from various device streams such as gaming events, IoT devices, and application streams. Pub/Sub ensures at-least-once delivery of received messages to subscribing applications, with no provisioning required. Pub/Sub's APIs are open, the service is global by default, and it offers end-to-end encryption.

Let's explore the end-to-end Pub/Sub architecture. Upstream source data comes in from devices all over the globe and is ingested into Pub/Sub, which is the first point of contact within the system. Pub/Sub reads, stores, and broadcasts to any subscribers of this data topic that new messages are available. As a subscriber of Pub/Sub, Dataflow can ingest and transform those messages in an elastic streaming pipeline and output the results into an analytics data warehouse like BigQuery. Finally, you can connect a data visualization tool, like Looker or Looker Studio, to visualize and monitor the results of a pipeline, or an AI or ML tool such as Vertex AI to explore the data to uncover business insights or help with predictions.

A central element of Pub/Sub is the topic. A topic is a named resource to which messages are sent by publishers. You can think of a topic like a radio antenna. Whether your radio is playing music or it's turned off, the antenna itself is always there. If music is being broadcast on a frequency that nobody's listening to, the stream of music still exists. Similarly, a publisher can send data to a topic that has no subscriber to receive it. Or a subscriber can be waiting for data from a topic that isn't getting data sent to it, like listening to static from a bad radio frequency. Or you could have a fully operational pipeline where the publisher is sending data to a topic that an application is subscribed to. That means there can be zero, one, or more publishers, and zero, one or more subscribers related to a topic. And they're completely decoupled, so they're free to break without affecting their counterparts.

It's helpful to describe this using an example. Say you've got a human resources topic. A new employee joins your company, and several applications across the company need to be updated. Adding a new employee can be an event that generates a notification to the other applications that are subscribed to the topic, and they'll receive the message about the new employee starting. Now, let's assume that there are two different types of employees: a full-time employee and a contractor. Both sources of employee data could have no knowledge of the other but still publish their events saying "this employee joined" into the Pub/Sub HR topic. After Pub/Sub receives the message, downstream applications like the directory service, facilities system, account provisioning, and badge activation systems can all listen and process their own next steps independent of one another.

Pub/Sub is a good solution to buffer changes for lightly coupled architectures, like this one, that have many different sources and sinks. Pub/Sub supports many different inputs and outputs, and you can even publish a Pub/Sub event from one topic to another. The next task is to get these messages reliably into our data warehouse, and we'll need a pipeline that can match Pub/Sub's scale and elasticity to do it.
