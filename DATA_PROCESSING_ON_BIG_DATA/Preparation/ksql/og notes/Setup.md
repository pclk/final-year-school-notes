# Confluent Kafka Community Setup Guide

## Learning Outcomes:

A. Setup Confluent Community Platform on local machine
B. Write a KSQL Command to create a stream

## Lab Environment:

This lab will be using Confluent Edition (6.1 and above) to setup the minimal platform for running Kafka and ksqlDB.

A Virtual Machine copy which is pre-installed with Ubuntu OS and Java 11 is being provided. The VM can be run using the VM Workstation in the lab.

The link to the VM is as follows:
https://drive.google.com/file/d/1IaIlAE7DDOOjsAon_mL0prnXa892f-oy/view?usp=share_link

Ubuntu User Name: Ubuntuser
Password: Password123

## Perform Update in Ubuntu:

Whenever you wish to perform any installation, remember to fire the following command in the terminal to get the latest update.

| Command             | Purpose                    |
| ------------------- | -------------------------- |
| sudo apt-get update | Fetch update software list |

## Practical Overview

A. Setting up Confluent on Local Machine
B. Verifying KSQLDB is up and running
C. Publishing a Kafka Topic

### A. Setting up Confluent on Local Machine

1. Update the Ubuntu Software List

2. Go to Confluent Platform (https://www.confluent.io/download) to download a copy of Confluent Local version. Select the TAR package to download. The file should be downloaded into the "Downloads" of the Ubuntu VM.

3. Go to Download folder. Enter the following command to uncompress the package:
   
   ```bash
   tar xfz confluent-7.4.0.tar.tar
   ```

4. Move the folder to the /opt folder:
   
   ```bash
   sudo mv confluent-7.4.0. /opt
   ```

5. Create a symbolic link to link Confluent to the folder:
   
   ```bash
   sudo ln -s confluent-7.4.0/ confluent
   ```

6. Setup the environment path to recognize Confluent and its binaries.
   Note: This step is crucial for calling confluent services from any directory. If a new terminal is created, ensure the path is still available.
   
   ```bash
   export PATH=${PATH}:/opt/confluent/bin
   export CONFLUENT_HOME=/opt/confluent
   ```

7. Start the Ksql-Server:
   
   ```bash
   confluent local services ksql-server start
   ```

### B. Verifying the ksqlDB is up and running

1. Enter the following command in terminal to start the ksql console. We term this console interface as [KSQL Console]:
   
   ```bash
   ksql
   ```

2. [SQL Console] Attempt to list the topics within the ksql console. You should be able to see a few system created topics:
   
   ```sql
   LIST TOPICS;
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-21-55-44-image.png) 

### C. Publishing Kafka Topic

1. Open a new terminal (so that you can concurrently view the KSQL console). We term this new terminal as [Confluent Terminal]. Enter the following command:
   
   ```bash
   kafka-topics --create --partitions 1 --replication-factor 1 --topic USERS --bootstrap-server localhost:9092
   ```
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-00-13-image.png)
   
   2. [KSQL Console] Enter the following commands in the KSQL Console:
   
   ```sql
   LIST TOPICS;
   PRINT 'USERS';
   ```
   
   PRINT "USERS"; will print nothing because the topic is empty - no data has been produced to it yet.
   
   3. [Confluent Terminal] Enter the following command to create a Producer:
   
   ```bash
   kafka-console-producer --broker-list localhost:9092 --topic USERS
   ```
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-06-19-image.png)
   
   4. [Confluent Terminal] Now, enter some usernames and their geographical location:
   
   ```
   Alice,SG
   Bob,US
   Charlie,MY
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-06-52-image.png)

## Try it yourself:

1. More KSQL Commands to try:
   
   ```sql
   LIST TOPICS;
   SHOW TOPICS;
   LIST STREAMS;
   SHOW STREAMS;
   PRINT 'USERS' FROM BEGINNING;
   PRINT 'USERS' FROM BEGINNING LIMIT 4;
   ```
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-08-37-image.png)
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-09-10-image.png)
   
   2. Create a Stream table out of the Topic 'USERS':
   
   ```sql
   CREATE STREAM users_stream (username VARCHAR, countrycode VARCHAR) 
   WITH (KAFKA_TOPIC='USERS', VALUE_FORMAT='DELIMITED');
   
   LIST STREAMS;
   SELECT username, countrycode FROM users_stream EMIT CHANGES;
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-09-59-image.png)

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-10-15-image.png)

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-10-39-image.png)
