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

## Google Cloud APIs

The third way to access Google Cloud is through application programming interfaces, or APIs. A software service's implementation can be complex and changeable. If each software service had to be coded for each implementation, the result would be brittle and error-prone.

Instead, application developers structure the software they write in a clean, well-defined interface that abstracts away needless detail, and then they document that interface. That's an Application Programming Interface. The underlying implementation can change as long as the interface doesn't, and other pieces of software that use the API don't have to know or care.

The services that make up Google Cloud offer APIs so that code you write can control them. The Google Cloud console includes a tool called the Google APIs Explorer that shows what APIs are available, and in what versions. Suppose you've explored an API, and you're ready to build an application that uses it. Do you have to start coding from scratch? No.

Google provides Cloud Client and Google API Client Libraries in many popular languages to take much of the drudgery out of the task of calling Google Cloud from your code. Languages currently represented in these libraries are: Java, Python, PHP, C##, Go, Node.js, Ruby and C++.

## Google Cloud Console App

The fourth and final way to access Google Cloud is through the Cloud Mobile App. The Cloud Mobile App provides a way for you to manage services running on Google Cloud directly from your mobile device. It's a convenient resource that comes at no extra cost.

The Cloud Mobile App can be used to start, stop, and use SSH to connect to Compute Engine instances and to see logs from each instance. It also lets you stop and start Cloud SQL instances. Additionally, you can administer applications deployed on App Engine by viewing errors, rolling back deployments, and changing traffic splitting.

The Cloud Mobile App provides up-to-date billing information for your projects and billing alerts for projects that are going over budget. You can set up customizable graphs showing key metrics such as CPU usage, network usage, requests per second, and server errors. The mobile app also offers alerts and incident management.

Download the Cloud Mobile App at cloud.google.com/console-app.
