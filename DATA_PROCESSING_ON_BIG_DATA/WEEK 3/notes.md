Question 1 (1 point)
You have executed the command 'gsutil acl ch -u AllUsers:R gs://bucket1/ada.jpg' in
Cloud Shell to make the file available to anyone. What visual indicator in the Storage
section of the Cloud Console allows you to verify that the permission level has been
set?

>> a) You see a public link to the image.

b) You see AllUsers under permissions.

> c) You see a green tick next to the bucket. 

d) You see public under storage class


Question 2 (1 point)
You want to create a bucket with a particular project name. You execute 'gsutil mb
gs://project.1' but receive the error message "BadRequestException' Why is this the
case?

a) Project.1 reveals sensitive information.

> b) The bucket name is already taken.

c) The bucket name cannot end with a number.

>> d) Project.1 is not a valid DNS name.


Question 3 (1 point)
Google Cloud has many storage options. Which would be the best option for
unstructured storage?

a) Cloud SQL

b) Cloud Bigtable

> c) Cloud Storage

d) Datastore








Question 4
Which of the following database engines does Cloud SQL support?

> MySQL
Oracle
MongoDB
> PostgreSQL
DB2

Question 5
What Cloud Storage storage class would be an inexpensive option for backups that
you test once a month?

> a) Nearline

b) Archive

c) Standard

d) Coldline


Question 6
You have analyzed your data and want to transfer relevant files to Cloud Storage.
Which data types are most suited for this type of storage?

Billing Information
Customer Transactions
Stock Information
> Images
> Documents

Question 7
You can use existing SQL skills to query data in Cloud Spanner.

> True
False

Question 8 (1 point)
Datastore is a NoSQL based managed service. Which statement best describes
Datastore?

> a) A document store
b) A sparse, wide-column database
c) A data warehouse
d) A relational database


Question 9 (1 point)
Cloud Bigtable provides a NoSQL based managed service option. Which one of the
following is a valid use case for this service?

> a) A backend for an Internet of Things (loT) system.

b) An object store.

c) A data warehouse.

d) A transactional database for a bank.


Question 10 (1 point)
You want to use the auto-prompting feature to help you learn the gcloud command
syntax. What actions must you take in the Cloud Shell in order to enable auto
prompting for commands and flags with inline help snippets?

a) Auto complete functionality is enabled by default.
b) Click on the Tab key twice to enable gcloud interactive mode.
> c) Change to your current working directory and run autocomplete.
>> d) Install the gcloud beta components and enter gcloud interactive mode.


Question 11
You want to add additional members to your project but are unable to do so. In the
Cloud Console in the cloud IAM & admin section you can see that you have the
roles/viewer permission. What role do you need in order to add a member to the
project?

>> a) roles/owner
> b) roles/editor
c) roles/custom
d) iam/serviceAccounts.actAs

Question 12 (1 point)
What is the purpose of a folder?

> a) Used to organize projects.
b) Used to organize users
c) Folders are not part of the hierarchy.
d) Used to organize instances.

Question 13 (1 point)
What does it mean for a system to be elastic?

a) The system is multi-cloud.
b) The system can be moved from region to region.
> c) The system can add and remove resources based on need.
d) The system can bounce back after an outage.


Question 14 
Which of the following best describes a major advantage Google Cloud has over
other cloud providers?

> a) Google owns one of the largest networks in the world. All regions and zones
are connected on the same network.
b) Google uses satellite links as a way of interconnecting Regions, Zones. and
POPS.
c) Google leases fiber from multiple vendors. In the event of a vendor's fiber
failing, another vendor's network can be used as a backup.
d) Google leverages traditional networking gear from commercial vendors.

Question 15 (1 point) Saved
Cloud Identity and Access Management (Cloud IAM) allows you to manage privileges.
What do these privileges apply to?

>> a) Google Cloud resources
> b) Applications, Google Cloud, and operating systems.
c) Applications.
d) Operating systems.

Question 16 (1 point)
Google Cloud provides a variety of service choices. Which of the following services is
infrastructure as a service (laaS)?

a) Google Kubernetes Engine
b) Cloud Function
> c) Compute Engine
d) App Engine

Question 17 (1 point)
How would you configure the Cloud SDK to work on your laptop?

a) Sync your laptop with Cloud Shell.
b) Download the config file from https://cloud.google.com.
> c) Run the gcloud init command.
d) Edit the .profile file of the SDK.

Question 18
Every Google Cloud service you use is associated with a project. Which of the
following statements regarding projects are correct? (Choose 2)

> A project cannot be restored once deleted.
Projects are billed and managed collectively.
>> Projects are the basis for enabling and using Google Cloud services.
Projects have a single owner.
>> Each project is a separate account, and each resource belongs to exactly one.

Question 19 (1 point)
The Cloud Shell code editor is one of the developer tools built into the Cloud
Console.

> True
False

Question 20
What best describes the purpose of quotas? (Choose 2)

Configuration used by Google to prevent building large systems.
> Configuration used to prevent over consumption due to malicious attacks.
Quotas are used to send billing alerts.
> Configuration used to prevent billing surprises.
Quotas Can automatically build infrastructure based on Cloud Logging metrics.


Question 21 (1 point)
How does Pub/Sub deliver messages to endpoints?

a) By pushing messages to all available endpoints based on an internal list.
> b) Through a publish/subscribe pattern.
c) Messages have to be pulled by the front end.
d) Through an internal ID.

Question 22 (1 point)
Which of the following is the most common API format in Google Cloud?

> a) REST
b) SOAP
c) gRPC
d) CLI

Question 23 (1 point)
You want to deploy and manage an API using Cloud Endpoints. Which application
management tasks will Cloud Endpoints help you with?

Application sizing
Asynchronous messaging
Data protection
> Interface definition
> Authentication and authorization


Question 24 (1 point)
You have a legacy backend application that you want to gradually move across to the
cloud and convert to microservices. What Google Cloud service can you use to
progressively convert the APIs from the backend service to the new cloud-based
microservices?

> a) Apigee Edge
b) Pub/Sub
c) Cloud Endpoints
d) Cloud Spanner


Question 25 (1 point)
You want to utilize Cloud Endpoints to create and manage your REST API. What
action must you take to achieve this?

a) Deploy the API's OpenAPI configuration using a service account to Google
Apps Script API.
> b) Deploy the API's OpenAPI configuration to Service Management.
c) Deploy the REST API configuration file to Firebase.
d) Create the configuration file using gRPC and upload to the Cloud SDK
Library.

Question 26 (1 point)
You need an application that will send notifications to remote applications that will
be offline for extended periods of time. What solution can you implement that will
guarantee that the remote applications will receive the notifications when they come
back online?

> a) Pub/Sub
b) Cloud Endpoint
c) Cloud Edge
d) Cloud API




Question 27 (1 point)
You are managing your API using Cloud Endpoints. The activity logs in Cloud Logging
indicate that a single client is excessively calling the API. What action can you take to
alleviate this?

a) Deploy Cloud Endpoints Frameworks and throttle the client.
b) Configure VPC quota limits and throttle client traffic.
> c) Deploy a Cloud Endpoints configuration that has a quota.
d) Request additional quota limits using the Cloud Console.

Question 28
You have to implement a solution that allows the Human Resources (HR) system to
reliably notify other departmental services that a new employee has been hired
without having to directly connect your application to all of the other services. What
Google service should you use?

a) Apigee Edge
> b) Pub/Sub
c) Dataflow
d) Cloud Connector

Question 29 (1 point)
You need to implement a big data analytics platform in the cloud. Into which phase
of the common big data processing model would you place Pub/Sub?

a) Analyze
b) Process
> c) Ingest
d) Store

Question 30 (1 point)
You want to utilize Cloud Endpoint to control access to your API. What actions can
you take to achieve this? (Choose all that applies)

Enable the Google APIs Explorer.
> Validate calls with JSON Web Tokens
Generate a SAML token.
> Generate and share API keys.
Deploy the Identity and Access Management API.

Question 3 (1 point)

You have non-relational data and want a serverless database without having to worry
about nodes or cluster management. Which service would best suit your needs?

a) Cloud Bigtable
> b) Datastore
c) Cloud SQL
d) Cloud Spanner


Question 4 (1 point)
Google Cloud has different options for SQL-based managed services. Which of these
options is horizontally scalable and globally available?

> a) Cloud Spanner
b) Datastore
c) Cloud Bigtable
d) Cloud SQL

Question 5 (1 point)
What is the largest object that you can store in Cloud Storage?

a) Dependant on the storage class
> b) 5 TB
c) 1 GB
d) Unlimited


Question 8 (1 point)
What type of service best describes Google Kubernetes Engine?

a) laaS
> b) PaaS
c) SaaS
d) Hybrid

Question 11 (1 point)
The Google Cloud hierarchy helps you manage resources across multiple
departments and multiple teams within an organization. Which of the following is at
the top level of this hierarchy?

a) Resource
> b) Organization
c) Project
d) Folder


Question 13 (1 point)
What are the types of message delivery supported with Pub/Sub? (Choose 2)

> Pull
Poll
Bounce
> Push

Question 14 (1 point)
You are creating a Cloud Endpoints configuration file for your API. What is the
unique identifier that you manually set which will be used to identify the name of the
service?

a) info.title
b) operationld
c) info.version
> d) host

Question 23 (1 point)
You want to ensure that files you are working on in the Cloud Shell persist across
multiple sessions. What action must you take to ensure this happens?

a) Archive your files to a persistent disk.
b) All saved files are persistent across sessions.
> c) Save the files in your home directory.
d) Export your files to Cloud Storage.


Question 26 (1 point)
You are deploying Cloud SQL. You need to gain management access to your Cloud
SQL instance from Cloud Shell. What action must you take?

> a) Allow the Cloud Shell instance IP address.
b) Configure management access using the GRANT statement.
c) Configure SSH access to the Cloud SQL instance.
d) Deny the Cloud Shell instance IP address.

Question 28 (1 point)
When using Cloud Shell you must also install the Cloud SDK locally to manage your
projects and resources.

> True
False

Question 30 (1 point)
How would you test a Google API and learn how it works?

> a) Use Google APIs Explorer that is part of the Cloud Console.
b) Use the help files in Cloud Shell.
c) Use the gcloud command in the Cloud SDK.
d) Use the console to get the directions on how to build the api by service.

Regions are independent geographic areas on the same continent. Which of the
following is a regional service?

a) HTTPS Load Balancer
>> b) Datastore
c) Network
> d) Virtual machine

Ģoogle Cloud provides resources in multi-regions, regions, and zones. What best
describes a zone?

> a) One or more data centers.
b) An edge location.
c) A point of presence (PoP).
d) Geographic location to leverage services.

You are configuring a Pub/Sub instance. What should a subscriber do when they
receive a message from a Subscription?

a) Acknowledge each message and move the message to the Ack store.
b) Acknowledge each message which marks the message as read.
c) Acknowledge each message and forward it on to other subscribers.
> d) Acknowledge each message within a configurable window of time.


How would you configure billing notifications in Google Cloud?

a) Use Cloud Functions to fire off an email with daily budget totals.
> b) Configure budgets and alerts.
c) Enable a script using cron to kick off when a threshold is reached.
d) Set up a billing alert in Cloud Monitoring.


In what format will a majority of the APIs return data in?

a) XML
b) TEXT
> c) JSON
d) YAML

Question 2 (1 point)
Projects form part of the Google Cloud resources hierarchy. Which of the following is
true concerning projects?

a) Projects are allocated a changeable Project ID.
b) Projects are only used for billing.
c) You can nest projects inside projects.
> d) All resources must be associated with a project.

Question 6
$3config is a command-line option for the Cloud SDK?

True
> False

Question 19 (1 point)
What is the purpose of an API?

a) APIs are non-HTTPS interfaces used to interface with web interfaces.
> b) APIs simplify the way disparate software resources communicate
c) APIs create GUI interfaces.
d) APIs replace web pages.

Question 29 (1 point)
You are using the Cloud Shell to create a virtual machine. You run the gcloud
compute command to create a virtual machine but omit the -- zone flag. What effect
will this have when provisioning the machine?

a) The virtual machine will be deployed in the nearest zone to your location.
b) Gcloud will prompt you to enter the zone information.
> c) Gcloud will infer your desired zone based on your default properties.
d) The virtual machine will be created initially but fail.

Which command line tool can be used to manage Cloud Storage?

a) bq
b) gcloud
> c) gsutil
d) Cloud Shell

Projects form part of the Google Cloud resources hierarchy. Which of the following is true concerning projects? 

a) Projects are allocated a changeable Project ID.
> b) All resources must be associated with a project.
c) You can nest projects inside projects.
d) Projects are only used for billing.


Cloud Identity and Access Management (Cloud IAM) allows you to manage privileges.
What do these privileges apply to?
> a) Google Cloud resources
b) Applications, Google Cloud, and operating systems.
c) Applications.
d) Operating systems.



Which of the following best describes a major advantage Google Cloud has over
other cloud providers?
> a) Google owns one of the largest networks in the world. All regions and zones
are connected on the same network.
b) Google uses satellite links as a way of interconnecting Regions, Zones. and
POPS.
c) Google leases fiber from multiple vendors. In the event of a vendor's fiber
failing, another vendor's network can be used as a backup.
d) Google leverages traditional networking gear from commercial vendors.


The Google Cloud hierarchy helps you manage resources across multiple
departments and multiple teams within an organization. Which of the following is at
the top level of this hierarchy?
a) Resource
> b) Organization
c) Project
d) Folder


You are deploying Cloud SQL. You need to gain management access to your Cloud
SQL instance from Cloud Shell. What action must you take?
> a) Allow the Cloud Shell instance IP address.
b) Configure management access using the GRANT statement.
c) Configure SSH access to the Cloud SQL instance.
d) Deny the Cloud Shell instance IP address.


Cloud Bigtable provides a NoSQL based managed service option. Which one of the
following is a valid use case for this service?
> a) A backend for an Internet of Things (loT) system.
b) An object store.
c) A data warehouse.
d) A transactional database for a bank.

Google Cloud has many storage options. Which would be the best option for
unstructured storage?
a) Cloud SQL
b) Cloud Bigtable
> c) Cloud Storage
d) Datastore

Google Cloud has different options for SQL-based managed services. Which of these
options is horizontally scalable and globally available?
> a) Cloud Spanner
b) Datastore
c) Cloud Bigtable
d) Cloud SQL


Datastore is a NoSQL based managed service. Which statement best describes
Datastore?
> a) A document store
b) A sparse, wide-column database
c) A data warehouse
d) A relational database



You want to use the auto-prompting feature to help you learn the gcloud command
syntax. What actions must you take in the Cloud Shell in order to enable auto
prompting for commands and flags with inline help snippets?
a) Auto complete functionality is enabled by default.
b) Click on the Tab key twice to enable gcloud interactive mode.
c) Change to your current working directory and run autocomplete.
> d) Install the gcloud beta components and enter gcloud interactive mode.


You are using the Cloud Shell to create a virtual machine. You run the gcloud
compute command to create a virtual machine but omit the -- zone flag. What effect
will this have when provisioning the machine?
a) The virtual machine will be deployed in the nearest zone to your location.
b) Gcloud will prompt you to enter the zone information.
> c) Gcloud will infer your desired zone based on your default properties.
d) The virtual machine will be created initially but fail.


How would you test a Google API and learn how it works?
> a) Use Google APIs Explorer that is part of the Cloud Console.
b) Use the help files in Cloud Shell.
c) Use the gcloud command in the Cloud SDK.
d) Use the console to get the directions on how to build the api by service.



How would you configure billing notifications in Google Cloud?
a) Use Cloud Functions to fire off an email with daily budget totals.
> b) Configure budgets and alerts.
c) Enable a script using cron to kick off when a threshold is reached.
d) Set up a billing alert in Cloud Monitoring.

What best describes the purpose of quotas? (Choose 2)
Configuration used by Google to prevent building large systems.
> Configuration used to prevent over consumption due to malicious attacks.
Quotas are used to send billing alerts.
> Configuration used to prevent billing surprises.
Quotas Can automatically build infrastructure based on Cloud Logging metrics.

Every Google Cloud service you use is associated with a project. Which of the
following statements regarding projects are correct? (Choose 2)
A project cannot be restored once deleted.
Projects are billed and managed collectively.
> Projects are the basis for enabling and using Google Cloud services.
Projects have a single owner.
> Each project is a separate account, and each resource belongs to exactly one.


You have to implement a solution that allows the Human Resources (HR) system to
reliably notify other departmental services that a new employee has been hired
without having to directly connect your application to all of the other services. What
Google service should you use?
a) Apigee Edge
> b) Pub/Sub
c) Dataflow
d) Cloud Connector

You are managing your API using Cloud Endpoints. The activity logs in Cloud Logging
indicate that a single client is excessively calling the API. What action can you take to
alleviate this?
a) Deploy Cloud Endpoints Frameworks and throttle the client.
b) Configure VPC quota limits and throttle client traffic.
> c) Deploy a Cloud Endpoints configuration that has a quota.
d) Request additional quota limits using the Cloud Console.


What is the purpose of an API?
a) APIs are non-HTTPS interfaces used to interface with web interfaces.
> b) APIs simplify the way disparate software resources communicate
c) APIs create GUI interfaces.
d) APIs replace web pages.

 
Google Cloud provides resources in multi-regions, regions, and zones. What best describes a zone?
Question 3 options:

a) A point of presence (PoP).

b) An edge location.

?> c) One or more data centers.

d) Geographic location to leverage services.

What Cloud Storage storage class would be an inexpensive option for backups that you test once a month?
a) Coldline
b) Archive
?> c) Nearline
d) Standard

