[toc]
# Week 1: GCCF: Cloud Computing Fundamentals

## Cloud Computing

Let's get started with cloud computing. The cloud is a hot topic these days, but what exactly is it?

The US National Institute of Standards and Technology created the term cloud computing, although there's nothing US-specific about it. Cloud computing is a way of using information technology, or IT, that has these five equally important traits.

First, customers get computing resources that are on demand and self-service. Through a web interface, users get the processing power, storage, and network they need with no need for human intervention.

Second, customers get access to those resources over the internet, from anywhere they have a connection. Third, the provider of those resources has a large pool of them and allocates them to users out of that pool. That allows the provider to buy in bulk and pass the savings onto the customers. Customers don't have to know or care about the exact physical location of those resources.

Fourth, the resources are elastic, which means that they can increase or decrease as needed, so customers can be flexible. If they need more resources, they can get more, and quickly. If they need less, they can scale back.

And finally, the customers pay only for what they use or reserve as they go. If they stop using resources, they stop paying. That's it. That's the definition of cloud computing.

An infrastructure is the basic underlying framework of facilities and systems. So it might be helpful to think about IT, or information technology, infrastructure in terms of a city's infrastructure. In a city, the infrastructure includes transportation, communications, power, water, fuel and other essential services.

Comparing it to IT infrastructure, the people in the city are like 'users,' and the cars, bikes, and buildings are like 'applications.' Everything that goes into creating and supporting those services is the infrastructure.

In this course, you'll explore the IT infrastructure services provided by Google Cloud. You'll become familiar enough with the infrastructure services to know what the services do, and you'll start to understand how to use them.

## Cloud vs Traditional Architecture

Now that you have a better understanding of what cloud computing is, and the infrastructure that supports it, let's transition to cloud architecture. In this section, we'll explore how the cloud compares to traditional architecture. To understand this, we need to look at some history.

The trend toward cloud computing started with a first wave known as colocation. Colocation gave users the financial efficiency of renting physical space, instead of investing in data center real estate. Virtualized data centers of today, which is the second wave, share similarities with the private data centers and colocation facilities of decades past. The components of virtualized data centers match the physical building blocks of hosted computing—servers, CPUs, disks, load balancers, and so on—but now they're virtual devices. With virtualization, enterprises still maintained the infrastructure; it's still a user-controlled and user-configured environment.

Several years ago, Google realized that its business couldn't move fast enough within the confines of the virtualization model. So Google switched to a container-based architecture—a fully automated, elastic third-wave cloud that consists of a combination of automated services and scalable data. Services automatically provision and configure the infrastructure used to run applications. Today, Google Cloud makes this third-wave cloud available to Google customers.

Google believes that, in the future, every company—regardless of size or industry—will differentiate itself from its competitors through technology. Increasingly, that technology will be in the form of software. Great software is based on high-quality data. This means that every company is, or will eventually become, a data company.

The virtual world, which includes Google Cloud's network, is built on physical infrastructure, and all those racks of humming servers use huge amounts of energy. Together, all existing data centers use roughly 2% of the world's electricity. So, Google works to make data centers run as efficiently as possible. Just like our customers, Google is trying to do the right things for the planet. We understand that Google Cloud customers have environmental goals of their own, and running their workloads in Google Cloud can be a part of meeting them.

Therefore, it's important to note that Google's data centers were the first to achieve ISO 14001 certification, which is a standard that maps out a framework for improving resource efficiency and reducing waste. This is Google's data center in Hamina, Finland. The facility is one of the most advanced and efficient data centers in the Google fleet. Its cooling system, which uses sea water from the Bay of Finland, reduces energy use and is the first of its kind anywhere in the world.

In our founding decade, Google became the first major company to be carbon neutral. In our second decade, we were the first company to achieve 100% renewable energy. By 2030, we aim to be the first major company to operate carbon free.

## IaaS, PaaS, and SaaS

Now let's shift our focus to IaaS, PaaS, and SaaS. The move to virtualized data centers introduced customers to two new types of offerings: Infrastructure as a Service, commonly referred to as IaaS, and Platform as a Service, or PaaS.

IaaS offerings provide raw compute, storage, and network capabilities organized virtually into resources that are similar to physical data centers. PaaS offerings bind code to libraries that provide access to the infrastructure applications need. This allows more resources to be focused on application logic.

In the IaaS model, customers pay for the resources they allocate ahead of time. In the PaaS model, customers pay for the resources they actually use. As cloud computing has evolved, the momentum has shifted toward managed infrastructure and managed services.

Leveraging managed resources and services allows companies to concentrate more on their business goals and spend less time and money on creating and maintaining their technical infrastructure. It allows companies to deliver products and services to their customers more quickly and reliably.

Serverless is yet another step in the evolution of cloud computing. Serverless computing allows developers to concentrate on their code rather than on server configuration by eliminating the need for any infrastructure management. Serverless technologies offered by Google include Cloud Functions, which manages event-driven code as a pay-as-you-go service, and Cloud Run, which allows customers to deploy their containerized microservices-based applications in a fully managed environment.

You might have also heard about Software as a Service, or SaaS, and wondered what it is and how it fits into the cloud ecosphere. SaaS applications are not installed on your local computer; they run in the cloud as a service and are consumed directly over the Internet by end users. Google's popular applications like Gmail, Docs, and Drive, collectively known as Google Workspace, are all classified as SaaS.

## Google Cloud Architecture

Next, let's focus on Google's specific offerings in the cloud. You can think of the Google Cloud infrastructure in three layers.

At the base layer is networking and security, which lays the foundation to support all of Google's infrastructure and applications. On the next layer sit compute and storage. Google Cloud separates, or decouples, as it's technically called, compute and storage so they can scale independently based on need.

And on the top layer sit the big data and machine learning products, which enable you to perform tasks to ingest, store, process, and deliver business insights, data pipelines, and machine learning models. And thanks to Google Cloud, you can accomplish these tasks without needing to manage and scale the underlying infrastructure.

Organizations with growing data needs often require lots of compute power to run big data jobs. And as organizations design for the future, the need for compute power only grows.

Google offers a range of computing services, which includes: Compute Engine, Google Kubernetes Engine, App Engine, Cloud Functions, Cloud Run. Google Cloud also offers a variety of managed storage options.

The list includes: Cloud Storage, Cloud SQL, Cloud Spanner, Cloud Bigtable, and Firestore. Cloud SQL and Cloud Spanner are relational databases, while Bigtable and Firestore are NoSQL databases.

And then there's a robust big data and machine learning product line. This includes: Cloud Storage, Dataproc, Bigtable, BigQuery, Dataflow, Firestore, Pub/Sub, Looker, Cloud Spanner, AutoML, and Vertex AI, the unified ML platform.

As we previously mentioned, the Google network is part of the foundation that supports all of Google's infrastructure and applications. Let's explore how that's possible.

Google's network is the largest network of its kind, and Google has invested billions of dollars over the years to build it. This network is designed to give customers the highest possible throughput and lowest possible latencies for their applications by leveraging more than 100 content caching nodes worldwide–locations where high demand content is cached for quicker access–to respond to user requests from the location that will provide the quickest response time.

Google Cloud's infrastructure is based in five major geographic locations: North America, South America, Europe, Asia, and Australia. Having multiple service locations is important because choosing where to locate applications affects qualities like availability, durability, and latency, which measures the time a packet of information takes to travel from its source to its destination.

Each of these locations is divided into several different regions and zones. Regions represent independent geographic areas, and are composed of zones. For example, London, or europe-west2, is a region that currently contains three different zones.

A zone is an area where Google Cloud resources are deployed. For example, let's say you launch a virtual machine using Compute Engine–more about Compute Engine in a bit–it will run in the zone that you specify to ensure resource redundancy. Zonal resources operate within a single zone, which means that if a zone becomes unavailable, the resources won't be available either.

Google Cloud lets users specify the geographical locations to run services and resources. In many cases, you can even specify the location on a zonal, regional, or multi-regional level. This is useful for bringing applications closer to users around the world, and also for protection in case there are issues with an entire region, say, due to a natural disaster.

A few of Google Cloud's services support placing resources in what we call a multi-region. For example, Cloud Spanner multi-region configurations allow you to replicate the database's data not just in multiple zones, but in multiple zones across multiple regions, as defined by the instance configuration. These additional replicas enable you to read data with low latency from multiple locations close to or within the regions in the configuration, like The Netherlands and Belgium.

Google Cloud currently supports 103 zones in 34 regions, though this is increasing all the time. The most up to date info can be found at cloud.google.com/about/locations.

## The Cloud Console

Let's begin with the Google Cloud console. There are actually four ways to access and interact with Google Cloud. The list includes the Google Cloud console, the Cloud SDK and Cloud Shell, the APIs, and the Cloud Mobile App. We'll explore all four of these options in this module, but focus on the console to start.

The Google Cloud console, which is Google Cloud's Graphical User Interface (GUI), helps you deploy, scale, and diagnose production issues in a simple web-based interface. With the console, you can easily find your resources, check their health, have full management control over them, and set budgets to control how much you spend on them.

The console also provides a search facility to quickly find resources and connect to instances through SSH, which is the Secure Shell Protocol, in the browser. To access the console, navigate to console.cloud.google.com.

## Understanding Projects

The console is used to access and use resources. Resources are organized in projects. To understand this organization, let's explore where projects fit in the greater Google Cloud resource hierarchy.

This hierarchy is made up of four levels, and starting from the bottom up they are: resources, projects, folders, and an organization node. At the first level are resources. These represent virtual machines, Cloud Storage buckets, tables in BigQuery, or anything else in Google Cloud.

Resources are organized into projects, which sit on the second level. Projects can be organized into folders, or even subfolders. These sit at the third level. And then at the top level is an organization node, which encompasses all the projects, folders, and resources in your organization.

Let's spend a little more time on the second level of the resource hierarchy, projects. Projects are the basis for enabling and using Google Cloud services, like managing APIs, enabling billing, adding and removing collaborators, and enabling other Google services.

Each project is a separate compartment, and each resource belongs to exactly one project. Projects can have different owners and users, because they're billed and managed separately.

Each Google Cloud project has three identifying attributes: a project ID, a project name, and a project number. The project ID is a globally unique identifier assigned by Google that cannot be changed–it is immutable–after creation. Project IDs are used in different contexts to inform Google Cloud of the exact project to work with.

The project names, however, are user-created. They don't have to be unique and they can be changed at any time, so they are not immutable. Google Cloud also assigns each project a unique project number. It's helpful to know that these Google-generated numbers exist, but we won't explore them much in this course. They are mainly used internally, by Google Cloud, to keep track of resources.

So, how are you expected to manage projects? Google Cloud has the Resource Manager tool, designed to programmatically help you do just that. It's an API that can gather a list of all the projects associated with an account, create new projects, update existing projects, and delete projects. It can even recover projects that were previously deleted and can be accessed through the RPC API and the REST API.

The third level of the Google Cloud resource hierarchy is folders. You can use folders to group projects under an organization in a hierarchy. For example, your organization might contain multiple departments, each with its own set of Google Cloud resources. Folders let you group these resources on a per-department basis. Folders give teams the ability to delegate administrative rights so that they can work independently.

To use folders, you must have an organization node, which is the topmost resource in the Google Cloud hierarchy. Everything else attached to that account goes under this node, which includes projects, folders, and other resources.

## Google Cloud Billing

The next topic is Google Cloud billing. Billing is established at the project level. This means that when you define a Google Cloud project, you link a billing account to it. This billing account is where you will configure all your billing information, including your payment option.

A billing account can be linked to zero or more projects, but projects that aren't linked to a billing account can only use free Google Cloud services. Billing accounts are charged automatically and invoiced every month or at every threshold limit. Billing sub-accounts can be used to separate billing by project. Some Google Cloud customers who resell Google Cloud services use sub-accounts for each of their own clients.

You're probably thinking, "How can I make sure I don't accidentally run up a big Google Cloud bill?" We provide a few tools to help:

1. You can define budgets at the billing account level or at the project level. A budget can be a fixed limit, or it can be tied to another metric - for example, a percentage of the previous month's spend.

2. To be notified when costs approach your budget limit, you can create an alert. For example, with a budget limit of $20,000 and an alert set at 90%, you'll receive a notification alert when your expenses reach $18,000. Alerts are generally set at 50%, 90%, and 100%, but can also be customized.

3. Reports is a visual tool in the Google Cloud console that lets you monitor expenditure based on a project or services.

4. Finally, Google Cloud also implements quotas, which are designed to prevent the over-consumption of resources because of an error or a malicious attack, protecting both account owners and the Google Cloud community as a whole.

There are two types of quotas: rate quotas and allocation quotas. Both are applied at the project level.

1. Rate quotas reset after a specific time. For example, by default, the GKE service implements a quota of 1,000 calls to its API from each Google Cloud project every 100 seconds. After that 100 seconds, the limit is reset.

2. Allocation quotas govern the number of resources you can have in your projects. For example, by default, each Google Cloud project has a quota allowing it no more than 5 Virtual Private Cloud networks.

Although projects all start with the same quotas, you can change some of them by requesting an increase from Google Cloud Support. If you're interested in estimating cloud computing costs on Google Cloud, you can try out the Google Cloud Pricing Calculator at cloud.google.com/products/calculator.

## Install and configure the Cloud SDK

Now let's explore the Cloud Software Development Kit (SDK), which lets users run Google Cloud command-line tools from a local desktop. The Cloud SDK is a set of command-line tools that you can use to manage resources and applications hosted on Google Cloud.

These include: the gcloud CLI, which provides the main command-line interface for Google Cloud products and services, gsutil (g-s-util), which lets you access Cloud Storage from the command line, and bq, a command-line tool for BigQuery.

When installed, all of the tools within the Cloud SDK are located under the bin directory. To install the Cloud SDK to your desktop, go to cloud.google.com/sdk and select the operating system for your desktop; this will download the SDK. Then follow the instructions specific to your operating system.

After the installation is complete, you'll need to configure the Cloud SDK for your Google Cloud environment. Run the gcloud init (gee-cloud in-it) command. You will be prompted for information including your login credentials, default project, and default region and zone.

## Cloud Shell

The next way to access and interact with Google Cloud is Cloud Shell. Cloud Shell provides command-line access to cloud resources directly from a browser. It's a Debian-based virtual machine with a persistent 5-GB home directory, which makes it easy to manage Google Cloud projects and resources.

With Cloud Shell, the Cloud SDK gcloud command and other utilities are always installed, available, up to date, and fully authenticated. To start Cloud Shell, navigate to console.cloud.google.com and click the Activate Cloud Shell icon on the toolbar. This will activate the Cloud Shell terminal, which will open in the lower portion of the window.

From the terminal window, you can launch the Cloud Shell code editor, which will open Cloud Shell in a new page. With the Cloud Shell code editor, you can edit files inside your Cloud Shell environment in real time within the web browser. This tool is convenient for working with code-first applications or container-based workloads, because you can easily edit files without needing to download and upload changes.

You can also use text editors from the Cloud Shell command prompt.

## Google Cloud APIs

The third way to access Google Cloud is through application programming interfaces, or APIs. A software service's implementation can be complex and changeable. If each software service had to be coded for each implementation, the result would be brittle and error-prone.

Instead, application developers structure the software they write in a clean, well-defined interface that abstracts away needless detail, and then they document that interface. That's an Application Programming Interface. The underlying implementation can change as long as the interface doesn't, and other pieces of software that use the API don't have to know or care.

The services that make up Google Cloud offer APIs so that code you write can control them. The Google Cloud console includes a tool called the Google APIs Explorer that shows what APIs are available, and in what versions. Suppose you've explored an API, and you're ready to build an application that uses it. Do you have to start coding from scratch? No.

Google provides Cloud Client and Google API Client Libraries in many popular languages to take much of the drudgery out of the task of calling Google Cloud from your code. Languages currently represented in these libraries are: Java, Python, PHP, C##, Go, Node.js, Ruby and C++.

## Google Cloud Mobile App

The fourth and final way to access Google Cloud is through the Cloud Mobile App. The Cloud Mobile App provides a way for you to manage services running on Google Cloud directly from your mobile device. It's a convenient resource that comes at no extra cost.

The Cloud Mobile App can be used to start, stop, and use SSH to connect to Compute Engine instances and to see logs from each instance. It also lets you stop and start Cloud SQL instances. Additionally, you can administer applications deployed on App Engine by viewing errors, rolling back deployments, and changing traffic splitting.

The Cloud Mobile App provides up-to-date billing information for your projects and billing alerts for projects that are going over budget. You can set up customizable graphs showing key metrics such as CPU usage, network usage, requests per second, and server errors. The mobile app also offers alerts and incident management.

Download the Cloud Mobile App at cloud.google.com/console-app.

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

# Week 5: Data Engineering - Building Batch Data Pipelines in Google Cloud Part 2

## Introduction to Dataflow

Let's start by exploring Dataflow in more detail. The reason Dataflow is the preferred way to do data processing on Google Cloud is that Dataflow is serverless. You don't have to manage clusters at all. Unlike with Dataproc, the auto scaling in Dataflow scales step by step, it's very fine-grained.

Plus, as we will see in the next course, Dataflow allows you to use the same code for both batch and stream. This is becoming increasingly important. When building a new data processing pipeline, we recommend that you use Dataflow.

If, on the other hand, you have existing pipelines written using Hadoop technologies, it may not be worth it to rewrite everything, migrate it over to Google Cloud using Dataproc and then modernize it as necessary. As a data engineer, we recommend that you learn both Dataflow and Dataproc and make the choice based on what's best for a specific use case. If the project has existing Hadoop or Spark dependencies, use Dataproc.

Please keep in mind that there are many subjective issues when making this decision, and that no simple guide will fit every use case. Sometimes the production team might be much more comfortable with a DevOps approach where they provision machines than with a serverless approach. In that case too, you might pick Dataproc. If you don't care about streaming and your primary goal is to move existing workloads, then Dataproc would be fine.

Dataflow, however, is our recommended approach for building pipelines. Dataflow provides a serverless way to execute pipelines on batch and streaming data, it's scalable to process more data, Dataflow will scale out to more machines, it will do this transparently. The stream processing capability also makes it low latency, you can process the data as it comes in.

This ability to process batch and stream with the same code is rather unique. For a long time, batch programming and data processing used to be two very separate and different things. Batch programming dates to the 1940s and the early days of computing where it was realized that you can think of two separate concepts, code and data. Use code to process data. Of course, both of these were on punch cards. So that's what you were processing, a box of punch cards called a batch. It was a job that started and ended when the data was fully processed.

Stream processing, on the other hand, is more fluid. It arose in the 1970s with the idea of data processing being something that is ongoing. The idea is that data keeps coming in and you process the data, the processing itself tended to be done in micro batches.

The genius of beam is that it provides abstractions that unify traditional batch programming concepts and traditional data processing concepts. Unifying programming and processing is a big innovation in data engineering. The four main concepts are P transforms, P collections, pipelines and pipeline runners.

A pipeline identifies the data to be processed and the actions to be taken on the data. The data is held in a distributed data abstraction called a P collection. The P collection is immutable. Any change that happens in a pipeline ingests one P collection and creates a new one. It does not change the incoming P collection.

The action or code is contained in an abstraction called a P transform. P transform handles input, transformation, and output of the data. The data in the P collection is passed along a graph from one P transform to another. Pipeline runners are analogous to container hosts such as Google Kubernetes Engine. The identical pipeline can be run on a local computer, data center VM, or on a service such as Dataflow in the Cloud. The only difference is scale and access to platform-specific services. The services the runner uses to execute the code is called a backend system.

Immutable data is one of the key differences between batch programming and data processing. Immutable data where each transform results in a new copy means there is no need to coordinate access control or sharing of the original ingest data. So it enables or at least simplifies distributed processing.

The shape of a pipeline is not actually just a singular linear progression but rather a directed graph with branches and aggregations. For historical reasons, we refer to it as a pipeline, but a data graph or Dataflow might be a more accurate description.

A P collection represents both streaming data and batch data. There is no size limit to a P collection. Streaming data is an unbounded P collection that doesn't end. Each element inside a P collection can be individually accessed and processed. This is how distributed processing of the P collection is implemented. So you define the pipeline and the transforms on the P collection and the runner handles implementing the transformations on each element distributing the work as needed for scale and with available resources.

Once an element is created in a P collection, it is immutable, so it can never be changed or deleted. Elements represent different data types. In traditional programs, a data type is stored in memory with a format that favors processing. Integers in memory are different from characters which are different from strings and compound data types.

In a P collection, all data types are stored in a serialized state as byte strings. This way, there is no need to serialize data prior to network transfer and deserialize it when it is received. Instead, the data moves through the system in a serialized state and is only deserialized when necessary for the actions of a P transform.


## Why customers value Dataflow

So we've discussed what Dataflow is, but why do data engineers value Dataflow over other alternatives for data processing? To understand that, it helps to understand a bit about how Dataflow works.

Dataflow provides an efficient execution mechanism for Apache Beam. The Beam pipeline specifies what has to be done. The Dataflow service chooses how to run the pipeline. The pipeline typically consists of reading data from one or more sources, applying processing to the data and writing it to one or more sinks.

In order to execute the pipeline, the Dataflow service first optimizes the graph by, for example, fusing transforms together. It then breaks the jobs into units of work and schedules them to various workers. One of the great things about Dataflow is that the optimization is always ongoing. Units of work are continually rebalanced.

Resources, both compute and storage, are deployed on demand and on a per job basis. Resources are torn down at the end of a job stage or on downscaling. Work scheduled on a resource is guaranteed to be processed. Work can be dynamically rebalanced across resources. This provides fault tolerance.

The watermarking handles late arrivals of data and comes with restarts, monitoring and logging. No more waiting for other jobs to finish. No more preemptive scheduling. Dataflow provides a reliable, serverless, job-specific way to process your data.

To summarize, the advantages of Dataflow are:
First, Dataflow is fully managed and auto-configured. Just deploy your pipeline.

Second, Dataflow doesn't just execute the Apache Beam transforms as is. It optimizes the graph, fusing operations, as we see with C and D. Also, it doesn't wait for a previous step to finish before starting a new step. We see this with A and the Group By Key.

Third, autoscaling happens step-by-step in the middle of a job. As the job needs more resources, it receives more resources. You don't have to manually scale resources to match job needs. If some machines have finished their tasks and others are still going on, the tasks queued up for the busy ones are rebalanced out to the idle machines. This way, the overall job finishes faster.

Dynamic work rebalancing in mid-job removes the need to spend operational or analyst resource time hunting down hotkeys. All this happens while maintaining strong streaming semantics. Aggregations, like sums and counts, are correct even if the input source sends duplicate records. Dataflow is able to handle late arriving records.

Finally, Dataflow functions as the glue that ties together many of the services on Google Cloud. Do you need to read from BigQuery and write to BigTable? Use Dataflow. Do you need to read from Pub/Sub and write to Cloud SQL? Use Dataflow.

## Building Dataflow pipelines in code

Let's look in greater detail at an example Dataflow pipeline. Here's how to construct a simple pipeline where you have an input PCollection and pass it through three PTransforms and get an output PCollection. The syntax is shown in Python.

You have the input, the pipe symbol, the first PTransform, the pipe symbol, the second PTransform, et cetera. The pipe operator essentially applies the transform to the input PCollection and sends out an output PCollection. The first three times, we don't give the output a name, simply pass it on the next step. The output of PTransform_3, though, we save into a PCollection variable named PCollection_out.

In Java, it is the same thing, except that, instead of the pipe symbol, we use the apply method. If you want to do branching, just send the same PCollection through two different transforms. Give the output PCollection variable in each case a name. Then you can use it in the remainder of your program.

Here, for example, we take the PCollection_in and pass the collection first through both PTransform_1 then through PTransform_2. The result of the first case, we store as PCollection_out_1. In the second case, we store it as PCollection_out_2.

What we showed you so far is the middle part of a pipeline. You already had a PCollection, and you applied a bunch of transforms, and you end up with a PCollection, but where does the pipeline start? How do you get the first PCollection? You get it from a source. What does a pipeline do with the final PCollection? Typically, it writes out to a sink. That's what we are showing here.

This is Python. We create a PCollection by taking the pipeline object P and passing it over a text file in cloud storage. That's the read from text line. Then, we apply the PTransform called FlatMap to the lines read from the text file. What FlatMap does is that it applies a function to each row of the input and concatenates all the outputs. When the function is applied to a row, it may return zero or more elements that go to the output PCollection.

The function in this case is the function called count_words. It takes a line of text and returns an integer. The output PCollection then consists of a set of integers. These integers are written to a text file in cloud storage. Because the pipeline was created in a with clause and because this is not a streaming pipeline, exiting the with clause automatically stops the pipeline.

Once you have written the pipeline, it's time to run it. Executing the Python program on the previous slide will run the program. By default, the program is run using the default runner, which runs on the same machine where the Python program was executed.

When you create the pipeline, you can pass in a set of options. One of these options is the runner. Specify that as Dataflow to have the pipeline run on Google Cloud. This example contains hard-coded variables, which in most cases is not a preferred practice for programming at scale. Of course, normally you will set up command-line parameters to transparently switch between local and cloud.

Simply running main runs the pipeline locally. To run on cloud, specify cloud parameters.

## Key considerations with designing pipelines

To design pipelines, you need to know how each step works on the individual data elements contained inside of a PCollection.

Let's start with the input and outputs of the pipeline. First, we set up our Beam pipeline with beam.Pipeline and pass through any options. Here, we'll call the pipeline P. Now it's time to get some data as input.

If we wanted to read a series of CSV files in Cloud Storage, we could use beam.io.ReadFromText and simply parse in the Cloud Storage bucket and file name. Note the use of an asterisk wildcard can handle multiple files.

If we wanted to read instead from a Pub/Sub topic, you would still use beam.io, but instead it's ReadStringsFromPubSub, and you'd have to parse in the topic name.

What about if you wanted to read in data that's already in BigQuery? Here's how that would look. You'd prepare your SQL query and specify BigQuery as your input source and then parse in the query and source as a read function to Dataflow.

These are just a few of the data sources from which Dataflow can read. But now what about writing to sinks?

Take the BigQuery example but as a data sink this time. With Dataflow, you can write to a BigQuery table, as you can see here. First, you establish the reference to the BigQuery table with what BigQuery expects: your project ID, dataset ID and table name. Then you use beam.io.WriteToBigQuery as a sink to your pipeline.

Note that we are using the normal BigQuery options here for write_disposition. Here, we're truncating the table if it exists, meaning to drop data rows. If the table doesn't exist, we can create it if needed.

Naturally, this is a batch pipeline if we're truncating the table with each load.

You can also create a PCollection in memory without reading from a particular source. Why might you do this? If you have a small dataset, like a lookup table or a hard-coded list, you could create the PCollection yourself, as you can see here. Then we can call a pipeline step on this new PCollection just as if we sourced it from somewhere else.

## Transforming data with PTransforms

Now that we have looked at how to get the data in, let's look at how we transform each data element in the PCollection with PTransforms.

The first step of any map-reduce process is the map phase, where you're doing something in parallel.

In the word length example, there is one length output for each word input, so the word "dog" would map to three for length.

In the bottom graph example, the function my_grep returns each instance of the term it's searching for in the line. There may be multiple instances of the term in a single line in a one-to-many relationship.

In this case, you may want my_grep to return the next instance each time it's called, which is why the function has been implemented with a generator using yields. The yield command has the effect of preserving the state of the function so that the next time it's called, it can continue from where it left off.

FlatMap has the effect of iterating over one-to-many relationships. The map example returns a key-value pair. In Python, this is simply a two-tuple for each word. The FlatMap example yields the line only for lines that contain the search term.

ParDo is a common intermediate step in a pipeline. You might use it to extract certain fields from a set of raw input records or convert raw input into a different format. You might also use ParDo to convert process data into an output format, like table rows for BigQuery or strings for printing.

You can use ParDo to consider each element in a PCollection and either output that element to a new collection or discard it. If your input PCollection contains elements that are of a different type or format than you want, you can use ParDo to perform a conversion on each element and output the result to a new PCollection.

If you have a PCollection of records with multiple fields, for example, you can use a ParDo to parse out just the fields you want to consider into a new PCollection. You can use ParDo to perform simple or complex computations on every element or certain elements of a PCollection and output the results as a new PCollection.

When you apply a ParDo transform, you need to provide code in the form of a DoFn object. A DoFn is a Beam SDK class that defines a distributed processing function. Your DoFn code must be fully serializable, idempotent and thread-safe.

In this example, we're just counting the number of words in a line and returning the length of the line. Transformations are always going to work on one element at a time here.

Here we have an example from Python which can return multiple variables. In this example, we have below and above, some cutoff in our data elements and return two different types, below and above, two different variables by referencing these properties of the results.

# Week 6: Data Engineering - Building Batch Data Pipelines in Google Cloud Part 3

## Introduction to Cloud Data Fusion

Let's start with an introduction to Cloud Data Fusion. Cloud Data Fusion provides a graphical user interface and APIs that increase time efficiency and reduce complexity. It equips business users, developers, and data scientists to quickly and easily build, deploy, and manage data integration pipelines.

Cloud Data Fusion is essentially a graphical no-code tool to build data pipelines. Cloud Data Fusion is used by developers, data scientists, and business analysts alike. For developers, Cloud Data Fusion allows you to cleanse, match, remove duplicates, blend, transform, partition, transfer, standardize, automate, and monitor data.

Data scientists can use Cloud Data Fusion to visually build integration pipelines, test, debug, and deploy applications. Business analysts can run Cloud Data Fusion at scale on Google Cloud, operationalize pipelines, and inspect rich integration metadata.

Cloud Data Fusion offers a number of benefits:
- Integrate with any data - through a rich ecosystem of connectors for a variety of legacy and modern systems, relational databases, file systems, cloud services, object stores, NoSQL, EBCDIC, and more.
- Increase productivity - If you have to constantly move between numerous systems to gather insight, your productivity is significantly reduced. With Cloud Data Fusion, your data from all the different sources can be pooled into a view like in BigQuery, Spanner, or any other Google Cloud technologies, allowing you to be more productive faster.
- Reduce complexity - through a visual interface for building data pipelines, code-free transformations, and reusable pipeline templates.
- Increase flexibility - through support for on-premises and cloud environments, interoperability with the open-source software CDAP.

At a high level, Cloud Data Fusion provides you with a graphical user interface to build data pipelines with no code. You can use existing templates, connectors to Google Cloud, and other Cloud services providers, and an entire library of transformations to help you get your data in the format and quality you want.

Also, you can test and debug the pipeline and follow along with each node as it receives and processes data. As you will see in the next lesson, you can tag pipelines to help organize them more efficiently for your team, and you can use the unified search functionality to quickly find field values or other keywords across your pipelines and schemas.

Lastly, we'll talk about how Cloud Data Fusion tracks the lineage of transformations that happen before and after any given field on your dataset. One of the advantages of Cloud Data Fusion is that it's extensible. This includes the ability to templatize pipelines, create conditional triggers, and manage and templatize plugins. There is a UI widget plugin as well as custom provisioners, custom compute profiles, and the ability to integrate to hubs.

## Components of Cloud Data Fusion

The two major user interface components we will focus our attention on in this course are the Wrangler UI for exploring data sets visually and building pipelines with no code, and the Data Pipeline UI for drawing pipelines right onto a canvas. You can choose from existing templates for common data processing tasks like Cloud Storage to BigQuery.

There are other features of Cloud Data Fusion that you should be aware of too. There's an integrated rules engine where business users can program in their pre-defined checks and transformations, and store them in a single place. Then data engineers can call these rules as part of a rule book or pipeline later.

We mentioned data lineage as part of field metadata earlier. You can use the metadata aggregator to access the lineage of each field in a single UI and analyze other rich metadata about your pipelines and schemas as well. For example, you can create and share a data dictionary for your schemas directly within the tool.

Other features, such as the microservice framework, allow you to build specialized logic for processing data. You can also use the Event Condition Action (ECA) Application to parse any event, trigger conditions, and execute an action based on those conditions.

## Cloud Data Fusion UI

Managing your pipelines is easiest when you have the right tools. We'll now take a high level look at the Cloud Data Fusion UI as you saw in the component overview. Here are some of the key user interface elements that you will encounter when using Data Fusion. Let's look at each of them in turn.

Under Control Center is the section for applications, artifacts and a dataset. Here you could have multiple pipelines associated with a particular application. The Control Center gives you the ability to see everything at a glance and search for what you need, whether it's a particular dataset, pipeline or other artifact, like a data dictionary, for example.

Under the Pipeline section, you have a developer studio. You can preview, export, schedule a job or project. You also have a connector and a function palette and a navigation section.

Under the Wrangler section, you have connections, transforms, data quality, insights and functions.

Under the Integration metadata section, you can search, add tags and properties and see the data lineage for field and data.

The Hub allows you to see all the available plugins, sample use cases and prebuilt pipelines.

Entities include the ability to create pipelines, upload an application, plugin, driver, library and directives.

There are two components in Administration, management and configuration. Under management, you have services and metrics. Under configuration, you have namespace, compute profiles, preferences, system artifacts and the REST client.

## Build a pipeline

Now that we've looked at the components in the UI, we'll discuss the process of building a data pipeline. A pipeline is represented visually as a series of stages arranged in a graph. These graphs are called DAGs, or Directed Acyclic Graphs, because they flow from one direction to another, and they cannot feed into themselves. Acyclic simply means not a circle.

Each stage is a node. And as you can see here, nodes can be of a different type, you may start with a node that pulls data from Cloud Storage, then passes it on to a node that parses a CSV. The next node takes multiple nodes, has an input and joins them together before passing the joined data to two separate data sink nodes.

As you saw in our previous example, you can have multiple nodes fork out from a single parent node. This is useful because you may want to kick off another data processing work stream that should not be blocked by any processing on a separate series of nodes. You can combine data from two or more nodes into a single output in a sink.

In Cloud Data Fusion, the studio is the user interface where you author and create new pipelines. The area where you create nodes and chain them together in your pipeline is your canvas. If you have many nodes in a pipeline, the canvas can get visually cluttered, so use the mini map to help navigate around a huge pipeline quickly. You can interact with the canvas and add objects by using the Canvas control panel.

When you're ready to save and run the entire pipeline, you can do so with the pipeline actions toolbar at the top. Don't forget to give your pipeline a name and description, as well as make use of the many pre-existing templates and plugins so you don't have to write your pipeline from scratch.

Here, we've used a template for data pipeline batch, which gives us the three nodes you see here to move data from a Cloud Storage file, process it in a wrangler and output it to BigQuery. You should make use of preview mode before you deploy and run your pipeline in production to ensure everything will run properly. While a pipeline is in preview, you can click on each node and see any sample data or errors that you will need to correct before deploying.

After deployment, you can monitor the health of your pipeline and collect key summary stats of each execution. Here, we're ingesting data from Twitter and Google Cloud and parsing each tweet before loading them into a variety of data sinks. If you have multiple pipelines, it's recommended that you make liberal use of the tags feature to help you quickly find and organize each pipeline for your organization.

You can view the start time, the duration of the pipeline run and the overall summary across runs for each pipeline. You can quickly see the data throughput at each node in the pipeline simply by interacting with the node. Note the compute profile used in the Cloud. Clicking on a node gives you detail on the inputs, outputs and errors for that given node.

Here, we are integrating with the speech-to-text API to process audio files into searchable text. You can track the individual health of each node and get useful metrics like records out per second, average processing time, and max processing time, which can alert you to any anomalies in your pipeline.

You can set your pipelines to run automatically at certain intervals. If your pipeline normally takes a long time to process the entire dataset, you can also specify a maximum number of concurrent runs to help avoid processing data unnecessarily. Keep in mind that Cloud Data Fusion is designed for batch data pipelines. We'll dive into streaming data pipelines in future modules.

One of the big features of Cloud Data Fusion is the ability to track the lineage of a given field value. Let's take this example of a campaign field for DoubleClick dataset and track every transform operation that happened before and after this field. Here, you can see the lineage of operations that are applied to the campaign field between the campaign dataset and the DoubleClick dataset. Note the time this field was last changed by a pipeline run and each of the input fields and descriptions that interacted with the field as part of processing it between datasets.

Imagine the use cases if you've inherited a set of analytical reports and you want to walk back upstream all of the logic that went into a certain field. Well, now you can.

## Explore data using wrangler

We've discussed the core components, tools and processes of building data pipelines. Now we'll look at using Wrangler to explore the data set. So far in the course, we have focused on building new pipelines for our data sets.

That presumes we know what the data is and what transformations need to be made already. Oftentimes, a new data set still needs to be explored and analyzed for insights. The Wrangler UI is the cloud data fusion environment for exploring new data sets visually for insights.

Here, you can inspect the data set and build a series of transformation steps called directives to stitch together a pipeline. Here's what the Wrangler UI looks like. Starting from the left, you have your connections to existing data sets.

You can add new connections to a variety of data sources like Google Cloud Storage, BigQuery or even other cloud providers. Once you specify your connection, you can browse all of the files and tables in that source. Here, you can see a cloud storage bucket of demo data sets and all the CSV files of customer complaints.

Once you've found an example data set like customers.csv here, you can explore the rows and columns visually and view sample insights. As you explore the data, you might want to create new calculated fields, drop columns, filter rows or otherwise wrangle the data.

You can do so using the Wrangler UI by adding new directives to form a data transformation recipe. When you're happy with your transformations, you can create a pipeline that you can then run at regular intervals.
