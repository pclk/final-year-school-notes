# Cloud computing
Let's get started with cloud computing.
00:03
The cloud is a hot topic these days, but what exactly is it?
00:07
The US National Institute of Standards and Technology created the term cloud computing, although there's nothing US-specific about it.
00:15
Cloud computing is a way of using information technology, or IT, that has these five equally important traits.
00:22
First, customers get computing resources that are on demand and self-service.
00:27
Through a web interface, users get the processing power, storage, and network they need with no need for human intervention.
00:36
Second, customers get access to those resources over the internet, from anywhere they have a connection.
00:43
Third, the provider of those resources has a large pool of them and allocates them to users out of that pool.
00:50
That allows the provider to buy in bulk and pass the savings onto the customers.
00:55
Customers don't have to know or care about the exact physical location of those resources.
01:01
Fourth, the resources are elastic, which means that they can increase or decrease as needed, so customers can be flexible.
01:09
If they need more resources, they can get more, and quickly.
01:13
If they need less, they can scale back.
01:16
And finally, the customers pay only for what they use or reserve as they go.
01:22
If they stop using resources, they stop paying.
01:26
That's it.
01:27
That's the definition of cloud computing.
01:30
An infrastructure is the basic underlying framework of facilities and systems.
01:34
So it might be helpful to think about IT, or information technology, infrastructure in terms of a city's infrastructure.
01:41
In a city, the infrastructure includes transportation, communications, power, water, fuel and other essential services.
01:50
Comparing it to IT infrastructure, the people in the city are like ‘users,’ and the cars, bikes,
01:55
and buildings are like ‘applications.’ Everything that goes into creating and supporting those services is the infrastructure.
02:03
In this course, you'll explore the IT infrastructure services provided by Google Cloud.
02:08
You'll become familiar enough with the infrastructure services to know what the services do, and you'll start to understand how to use them.

# Cloud vs traditional architecture
Now that you have a better understanding of what cloud computing is, and the infrastructure that supports it, let’s transition to cloud architecture.
00:08
In this section, we’ll explore how the cloud compares to traditional architecture.
00:13
To understand this, we need to look at some history.
00:16
The trend toward cloud computing started with a first wave known as colocation.
00:23
Colocation gave users the financial efficiency of renting physical space, instead of investing in data center real estate.
00:31
Virtualized data centers of today, which is the second wave, share similarities with the private data centers and colocation facilities of decades past.
00:39
The components of virtualized data centers match the physical building blocks of hosted computing—servers, CPUs, disks, load balancers, and so on—but now they’re virtual devices.
00:52
With virtualization, enterprises still maintained the infrastructure; it’s still a user-controlled and user-configured environment.
01:01
Several years ago, Google realized that its business couldn’t move fast enough within the confines of the virtualization model.
01:08
So Google switched to a container-based architecture—a fully automated, elastic third-wave cloud that consists of a combination of automated services and scalable data.
01:20
Services automatically provision and configure the infrastructure used to run applications.
01:25
Today, Google Cloud makes this third-wave cloud available to Google customers.
01:31
Google believes that, in the future, every company—regardless of size or industry—will differentiate itself from its competitors through technology.
01:39
Increasingly, that technology will be in the form of software.
01:44
Great software is based on high-quality data.
01:47
This means that every company is, or will eventually become, a data company.
01:53
The virtual world, which includes Google Cloud’s network, is built on physical infrastructure, and all those racks of humming servers use huge amounts of energy.
02:03
Together, all existing data centers use roughly 2% of the world’s electricity.
02:09
So, Google works to make data centers run as efficiently as possible.
02:14
Just like our customers, Google is trying to do the right things for the planet.
02:18
We understand that Google Cloud customers have environmental goals of their own, and running their workloads in Google Cloud can be a part of meeting them.
02:25
Therefore, it’s important to note that Google's data centers were the first to achieve ISO 14001
02:31
certification, which is a standard that maps out a framework for improving resource efficiency and reducing waste.
02:39
This is Google’s data center in Hamina, Finland.
02:41
The facility is one of the most advanced and efficient data centers in the Google fleet.
02:47
Its cooling system, which uses sea water from the Bay of Finland, reduces energy use and is the first of its kind anywhere in the world.
02:56
In our founding decade, Google became the first major company to be carbon neutral.
03:01
In our second decade, we were the first company to achieve 100% renewable energy.
03:06
By 2030, we aim to be the first major company to operate carbon free.

# IaaS, PaaS, and SaaS
now let's shift our Focus to is pass and
SAS
the move to virtualized Data Centers
introduce customers to two new types of
offerings
infrastructure as a service commonly
referred to as is and platform as a
service or pass
is offerings provide raw compute storage
and network capabilities
organized virtually into resources that
are similar to physical data centers
pass offerings bind code to libraries
that provide access to the
infrastructure applications need
this allows more resources to be focused
on application logic
in the is model customers pay for the
resources they allocate ahead of time
in the past model customers pay for the
resources they actually use
as cloud computing has evolved the
momentum has shifted toward managed
infrastructure and managed services
leveraging managed resources and
services allows companies to concentrate
more on their business goals and spend
less time and money on creating and
maintaining their technical
infrastructure
it allows companies to deliver products
and services to their customers more
quickly and reliably
serverless is yet another step in the
evolution of cloud computing
serverless Computing allows developers
to concentrate on their code rather than
on server configuration by eliminating
the need for any infrastructure
management
serverless Technologies offered by
Google include Cloud functions which
manages event-driven code as a
pay-as-you-go service
and Cloud run which allows customers to
deploy their containerized microservices
based application in a fully managed
environment
you might have also heard about software
as a service or SAS and wondered what it
is and how it fits into the cloud
ecosphere
SAS applications are not installed on
your local computer they run in the
cloud as a service and are consumed
directly over the Internet by end users
Google's popular applications like Gmail
docs and drive collectively known as
Google workspace are all classified as
SAS

# Google Cloud Architecture
Next, let's focus on Google’s specific offerings in the cloud.
00:04
You can think of the Google Cloud infrastructure in three layers.
00:08
* At the base layer is networking and security, which lays the foundation to support all of Google’s infrastructure and applications.
00:15
* On the next layer sit compute and storage.
00:19
Google Cloud separates, or decouples, as it’s technically called, compute and storage so they can scale independently based on need.
00:26
* And on the top layer sit the big data and machine learning products, which enable you
00:31
to perform tasks to ingest, store, process, and deliver business insights, data pipelines, and machine learning models.
00:41
And thanks to Google Cloud, you can accomplish these tasks without needing to manage and scale the underlying infrastructure.
00:49
Organizations with growing data needs often require lots of compute power to run big data jobs.
00:55
And as organizations design for the future, the need for compute power only grows.
01:01
Google offers a range of computing services, which includes: * Compute Engine * Google Kubernetes Engine *
01:08
App Engine * Cloud Functions * Cloud Run Google Cloud also offers a variety of managed storage options.
01:18
The list includes: * Cloud Storage * Cloud SQL * Cloud Spanner * Cloud Bigtable, and
01:26
* Firestore Cloud SQL and Cloud Spanner are relational databases, while Bigtable and Firestore are NoSQL databases.
01:37
And then there’s a robust big data and machine learning product line.
01:40
This includes: * Cloud Storage * Dataproc * Bigtable * BigQuery * Dataflow * Firestore * Pub/Sub * Looker * Cloud Spanner * AutoML, and *
01:56
Vertex AI, the unified ML platform As we previously mentioned, the Google network is part of the foundation that supports all of Google’s infrastructure and applications.
02:07
Let’s explore how that’s possible.
02:10
Google’s network is the largest network of its kind, and Google has invested billions of dollars over the years to build it.
02:18
This network is designed to give customers the highest possible throughput and lowest possible latencies for their applications by leveraging more than 100 content caching
02:26
nodes worldwide–locations where high demand content is cached for quicker access–to respond to user requests from the location that will provide the quickest response time.
02:38
Google Cloud’s infrastructure is based in five major geographic locations: North America, South America, Europe, Asia, and Australia.
02:50
Having multiple service locations is important because choosing where to locate applications affects qualities like availability, durability, and
02:56
latency, which measures the time a packet of information takes to travel from its source to its destination.
03:04
Each of these locations is divided into several different regions and zones.
03:11
Regions represent independent geographic areas, and are composed of zones.
03:16
For example, London, or europe-west2, is a region that currently contains three different zones.
03:23
A zone is an area where Google Cloud resources are deployed.
03:28
For example, let’s say you launch a virtual machine using Compute Engine–more about Compute Engine in a bit–it will run in the zone that you specify to ensure resource redundancy.
03:39
Zonal resources operate within a single zone, which means that if a zone becomes unavailable, the resources won’t be available either.
03:48
Google Cloud lets users specify the geographical locations to run services and resources.
03:54
In many cases, you can even specify the location on a zonal, regional, or multi-regional level.
04:01
This is useful for bringing applications closer to users around the world, and also for
04:05
protection in case there are issues with an entire region, say, due to a natural disaster.
04:11
A few of Google Cloud’s services support placing resources in what we call a multi-region.
04:18
For example, Cloud Spanner multi-region configurations allow you to replicate the database's data not just
04:24
in multiple zones, but in multiple zones across multiple regions, as defined by the instance configuration.
04:32
These additional replicas enable you to read data with low latency from multiple locations close to or within the regions in the configuration, like The Netherlands and Belgium.
04:42
Google Cloud currently supports 103 zones in 34 regions, though this is increasing all the time.
04:49
The most up to date info can be found at cloud.google.com/about/locations.

# The cloud console
Let’s begin with the Google Cloud console.
00:03
There are actually four ways to access and interact with Google Cloud.
00:07
The list includes the Google Cloud console, the Cloud SDK and Cloud Shell, the APIs, and the Cloud Mobile App.
00:16
We’ll explore all four of these options in this module, but focus on the console to start.
00:22
The Google Cloud console, which is Google Cloud’s Graphical User Interface (GUI), helps you deploy, scale, and diagnose production issues in a simple web-based interface.
00:35
With the console, you can easily find your resources, check their health, have full management control over them, and set budgets to control how much you spend on them.
00:43
The console also provides a search facility to quickly find resources and connect to instances through SSH, which is the Secure Shell Protocol, in the browser.
00:56
To access the console, navigate to console.cloud.google.com.

# Understanding projects
The console is used to access and use resources.
00:05
Resources are organized in projects.
00:07
To understand this organization, let’s explore where projects fit in the greater Google Cloud resource hierarchy.
00:14
This hierarchy is made up of four levels, and starting from the bottom up they are: resources, projects, folders, and an organization node.
00:25
At the first level are resources.
00:29
These represent virtual machines, Cloud Storage buckets, tables in BigQuery, or anything else in Google Cloud.
00:38
Resources are organized into projects, which sit on the second level.
00:43
Projects can be organized into folders, or even subfolders.
00:46
These sit at the third level.
00:49
And then at the top level is an organization node, which encompasses all the projects, folders, and resources in your organization.
00:57
Let’s spend a little more time on the second level of the resource hierarchy, projects.
01:03
Projects are the basis for enabling and using Google Cloud services, like managing APIs, enabling billing, adding and removing collaborators, and enabling other Google services.
01:17
Each project is a separate compartment, and each resource belongs to exactly one project.
01:24
Projects can have different owners and users, because they’re billed and managed separately.
01:29
Each Google Cloud project has three identifying attributes: a project ID, a project name, and a project number.
01:37
* The project ID is a globally unique identifier assigned by Google that cannot be changed–it is immutable–after creation.
01:46
Project IDs are used in different contexts to inform Google Cloud of the exact project to work with.
01:53
* The project names, however, are user-created.
01:55
They don’t have to be unique and they can be changed at any time, so they are not immutable.
02:03
* Google Cloud also assigns each project a unique project number.
02:06
It’s helpful to know that these Google-generated numbers exist, but we won’t explore them much in this course.
02:12
They are mainly used internally, by Google Cloud, to keep track of resources.
02:17
So, how are you expected to manage projects?
02:22
Google Cloud has the Resource Manager tool, designed to programmatically help you do just that.
02:27
It’s an API that can gather a list of all the projects associated with an account, create new projects, update existing projects, and delete projects.
02:37
It can even recover projects that were previously deleted and can be accessed through the RPC API and the REST API.
02:44
The third level of the Google Cloud resource hierarchy is folders.
02:49
You can use folders to group projects under an organization in a hierarchy.
02:54
For example, your organization might contain multiple departments, each with its own set of Google Cloud resources.
03:01
Folders let you group these resources on a per-department basis.
03:06
Folders give teams the ability to delegate administrative rights so that they can work independently.
03:13
To use folders, you must have an organization node, which is the topmost resource in the Google Cloud hierarchy.
03:19
Everything else attached to that account goes under this node, which includes projects, folders, and other resources.

# Google cloud billing
The next topic is Google Cloud billing.
00:02
Billing is established at the project level.
00:06
This means that when you define a Google Cloud project, you link a billing account to it.
00:11
This billing account is where you will configure all your billing information, including your payment option.
00:17
A billing account can be linked to zero or more projects, but projects that aren’t linked to a billing account can only use free Google Cloud services.
00:28
Billing accounts are charged automatically and invoiced every month or at every threshold limit.
00:34
Billing sub accounts can be used to separate billing by project.
00:38
Some Google Cloud customers who resell Google Cloud services use sub accounts for each of their own clients.
00:44
You’re probably thinking, “How can I make sure I don’t accidentally run up a big Google Cloud bill?”
00:50
We provide a few tools to help.
00:52
1. You can define budgets at the billing account level or at the project level.
00:58
A budget can be a fixed limit, or it can be tied to another metric - for example, a percentage of the previous month’s spend.
01:05
2. To be notified when costs approach your budget limit, you can create an alert.
01:11
For example, with a budget limit of $20,000 and an alert set at 90%, you’ll receive a notification alert when your expenses reach $18,000.
01:22
Alerts are generally set at 50%, 90% and 100%, but can also be customized.
01:29
3. Reports is a visual tool in the Google Cloud console that lets you monitor expenditure based on a project or services.
01:37
4. Finally, Google Cloud also implements quotas, which are designed to prevent the over-consumption of resources because of
01:43
an error or a malicious attack, protecting both account owners and the Google Cloud community as a whole.
01:49
There are two types of quotas: rate quotas and allocation quotas.
01:55
Both are applied at the project level.
01:58
1. Rate quotas reset after a specific time.
02:01
For example, by default, the GKE service implements a quota of 1,000 calls to its API from each Google Cloud project every 100 seconds.
02:12
After that 100 seconds, the limit is reset.
02:15
2. Allocation quotas govern the number of resources you can have in your projects.
02:20
For example, by default, each Google Cloud project has a quota allowing it no more than 5 Virtual Private Cloud networks.
02:29
Although projects all start with the same quotas, you can change some of them by requesting an increase from Google Cloud Support.
02:36
If you’re interested in estimating cloud computing costs on Google Cloud, you can try out the Google Cloud Pricing Calculator at cloud.google.com/products/calculator.

# Install and configure the Cloud SDK
Now let’s explore the Cloud Software Development Kit (SDK), which lets users run Google Cloud command-line tools from a local desktop.
00:09
The Cloud SDK is a set of command-line tools that you can use to manage resources and applications hosted on Google Cloud.
00:18
These include: the gcloud CLI, which provides the main command-line interface for Google Cloud products and services, gsutil
00:24
(g-s-util), which lets you access Cloud Storage from the command line, and bq, a command-line tool for BigQuery.
00:35
When installed, all of the tools within the Cloud SDK are located under the bin directory.
00:42
To install the Cloud SDK to your desktop, go to cloud.google.com/sdk and select the operating system for your desktop; this will download the SDK.
00:52
Then follow the instructions specific to your operating system.
00:58
After the installation is complete, you’ll need to configure the Cloud SDK for your Google Cloud environment.
01:04
Run the gcloud init (gee-cloud in-it) command.
01:06
You will be prompted for information including your login credentials, default project, and default region and zone.

# Cloud Shell
The next way to access and interact with Google Cloud is Cloud Shell.
00:04
Cloud Shell provides command-line access to cloud resources directly from a browser.
00:09
It’s a Debian-based virtual machine with a persistent 5-GB home directory, which makes it easy to manage Google Cloud projects and resources.
00:20
With Cloud Shell, the Cloud SDK gcloud command and other utilities are always installed, available, up to date, and fully authenticated.
00:29
To start Cloud Shell, navigate to console.cloud.google.com and click the Activate Cloud Shell icon on the toolbar.
00:39
This will activate the Cloud Shell terminal, which will open in the lower portion of the window.
00:46
From the terminal window, you can launch the Cloud Shell code editor, which will open Cloud Shell in a new page.
00:53
With the Cloud Shell code editor, you can edit files inside your Cloud Shell environment in real time within the web browser.
01:01
This tool is convenient for working with code-first applications or container-based workloads, because you can easily edit files without needing to download and upload changes.
01:11
You can also use text editors from the Cloud Shell command prompt.

# Google Cloud APIs
The third way to access Google Cloud is through application programming interfaces, or APIs.
00:07
A software service’s implementation can be complex and changeable.
00:11
If each software service had to be coded for each implementation, the result would be brittle and error-prone.
00:18
So instead, application developers structure the software they write in a clean, well-defined interface that abstracts away needless detail, and then they document that interface.
00:28
That’s an Application Programming Interface.
00:31
The underlying implementation can change as long as the interface doesn’t, and other pieces of software that use the API don’t have to know or care.
00:41
The services that make up Google Cloud offer APIs so that code you write can control them.
00:47
The Google Cloud console includes a tool called the Google APIs Explorer that shows what APIs are available, and in what versions.
00:57
Suppose you’ve explored an API, and you’re ready to build an application that uses it.
01:01
Do you have to start coding from scratch?
01:04
No.
01:05
Google provides Cloud Client and Google API Client Libraries in many popular languages to take much of the drudgery out of the task of calling Google Cloud from your code.
01:16
Languages currently represented in these libraries are: Java, Python, PHP, C#, Go, Node.js, Ruby and C++.

# Google Cloud console app
The fourth and final way to access Google Cloud is through the Cloud Mobile App.
00:05
The Cloud Mobile App provides a way for you to manage services running on Google Cloud directly from your mobile device.
00:11
It’s a convenient resource that comes at no extra cost.
00:15
The Cloud Mobile App can be used to start, stop, and use ssh to connect to Compute Engine instances and to see logs from each instance.
00:24
It also lets you stop and start Cloud SQL instances.
00:28
Additionally, you can administer applications deployed on App Engine by viewing errors, rolling back deployments, and changing traffic splitting.
00:37
The Cloud Mobile App provides up-to-date billing information for your projects and billing alerts for projects that are going over budget.
00:45
You can set up customizable graphs showing key metrics such as CPU usage, network usage, requests per second, and server errors.
00:53
The mobile app also offers alerts and incident management.
00:57
Download the Cloud Mobile App at cloud.google.com/console-app.
