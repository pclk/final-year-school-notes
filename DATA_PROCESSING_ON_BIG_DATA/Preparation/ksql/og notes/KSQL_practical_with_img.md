# Confluent Kafka: Introduction and Environment Setup

## Background

Confluent Kafka is a powerful platform for real-time data streaming and processing. The combination of Datagen and KsqlDB provides developers with tools to generate test data and manipulate streams using SQL-like syntax, making it easier to prototype and validate streaming applications.

## Lab Environment Setup

### 1. Terminal Configuration

You'll need to open two terminals:

**Terminal 1 (Ksql Terminal)**

```bash
confluent local services status
confluent local services ksql-server start
```

**Terminal 2 (Confluent Terminal)**

```bash
echo $PATH
echo $CONFLUENT_HOME
```

Why do we need two separate terminals?

Two terminals are required because:

1. The KSQL terminal is dedicated to running the KSQL server and executing queries
2. The Confluent terminal is used for data generation and other Kafka operations

This separation allows for better monitoring and management of different components.

What are the key environment variables needed?

The essential environment variables are:

1. CONFLUENT_HOME: Points to the Confluent installation directory
2. PATH: Must include the Confluent binary directory for accessing commands

These variables ensure proper access to Confluent tools and utilities.

What services should be running?

The essential services include:

1. Zookeeper
2. Kafka broker
3. Schema Registry
4. KSQL Server

All these services must be running for the lab exercises to work correctly.

# Confluent Kafka: Basic Stream Creation

## Creating Orders Stream

### 1. Generating Orders Data

```bash
# Confluent Terminal
ksql-datagen quickstart=orders topic=orders_topic
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-50-34-image.png)

### 2. Creating Orders Stream Structure

```sql
# Ksql Terminal
list topics;

CREATE STREAM orders_raw (
    orderid varchar key,
    orderunits double as price,
    address struct
        city varchar,
        state varchar,
        zipcode int
    >,
    ordertime VARCHAR
) WITH (
    KAFKA_TOPIC='orders_topic',
    VALUE_FORMAT='JSON'
);
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-51-42-image.png)

```sql
DESCRIBE orders_raw;
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-52-01-image.png)

Why use a struct for address?

A struct data type is used for the address because it:

1. Groups related fields logically

2. Maintains data hierarchy

3. Makes it easier to query and manipulate nested data

## Creating Users Stream

### 1. Generating Users Data

```bash
# Confluent Terminal
ksql-datagen quickstart=users topic='users_kafka_topic_json'
```

### 2. Creating Users Table

```sql
CREATE TABLE users_original (
    userid VARCHAR PRIMARY KEY,
    registertime BIGINT,
    gender VARCHAR,
    regionid VARCHAR
) WITH (
    kafka_topic='users_kafka_topic_json',
    value_format='JSON'
);
```

What's the difference between STREAM and TABLE?

- STREAM: Represents an unbounded sequence of events, where each record is independent

- TABLE: Represents the current state of entities, where each key has only one current value

## Creating Pageviews Stream

### 1. Generating Pageviews Data

```bash
# Confluent Terminal
ksql-datagen quickstart=pageviews topic=pageviews
```

### 2. Creating Pageviews Stream Structure

```sql
CREATE STREAM pageviews_original (
    viewtime bigint,
    userid varchar,
    pageid varchar
) WITH (
    kafka_topic='pageviews',
    value_format='json'
);
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-53-16-image.png)

Why use BIGINT for viewtime?

BIGINT is used for viewtime because:

1. It can store large timestamp values
2. Provides sufficient range for Unix timestamps
3. Supports time-based operations and windowing

## Generating Dummy Stream using Built-in Schema

This exercise simulates complex data creation that involves "Array".

KsqlDB enables using complex data types, like arrays and maps, in your queries. The syntax would be like `myarray ARRAY<type>` and using indexing `myarray[0]` to declare and access these types.

1. [Confluent Terminal] Generate dummy stream using the built-in schema:
   
   ```
   ksql-datagen quickstart=users topic=users_extended
   ```

2. [Ksql Terminal] List the Topic followed by create a stream:
   
   ```sql
   CREATE TABLE users_extended (
     userid VARCHAR PRIMARY KEY,
     registertime BIGINT,
     gender VARCHAR,
     regionid VARCHAR,
     interests ARRAY<STRING>,
     contactInfo MAP<STRING, STRING>
   ) WITH (
     kafka_topic='users_extended',
     value_format='JSON'
   );
   ```

3. ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-53-45-image.png)
   
   [Ksql Terminal] Describe the stream:
   
   ```sql
   DESCRIBE users_extended;
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-54-09-image.png) 

# Confluent Kafka: Download schema.

2. [Confluent Terminal] Run the following command to download the avro
   
   ```bash
   curl -o impressions.avro https://raw.githubusercontent.com/apurvam/streams-prototyping/master/src/main/resources/impressions.avro
   ```

3. [Confluent Terminal] Generate the Custom made schema
   
   ```bash
   ksql-datagen schema=~/impressions.avro format=delimited topic=impressions key=impressionid
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-56-05-image.png) 

## Sample data:

```json
{
  "namespace": "streams",
  "name": "impressions",
  "type": "record",
  "fields": [
    {
      "name": "impresssiontime",
      "type": {
        "type": "long",
        "format_as_time": "unix_long",
        "arg.properties": {
          "iteration": {
            "start": 1,
            "step": 10
          }
        }
      }
    },
    {
      "name": "impressionid",
      "type": {
        "type": "string",
        "arg.properties": {
          "regex": "impression_[1-9][0-9][0-9]"
        }
      }
    },
    {
      "name": "userid",
      "type": {
        "type": "string",
        "arg.properties": {
          "regex": "user_[1-9][0-9]?"
        }
      }
    },
    {
      "name": "adid",
      "type": {
        "type": "string",
        "arg.properties": {
          "regex": "ad_[1-9][0-9]?"
        }
      }
    }
  ]
}
```

4. [Ksql Terminal] List the Topic:
   
   ```sql
   list topics;
   ```

5. [Ksql Terminal] Create a "impressions" stream:
   
   ```sql
   CREATE STREAM impressions 
   (viewtime BIGINT, 
    key VARCHAR, 
    userid VARCHAR,
    adid VARCHAR)
   WITH (KAFKA_TOPIC='impressions', 
         VALUE_FORMAT='DELIMITED');
   ```

6. ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-54-32-image.png)
   
   [Ksql Terminal] Create a second stream based on the impressions stream:
   
   ```sql
   CREATE STREAM impressions2 AS
   SELECT * FROM impressions
   EMIT CHANGES;
   ```

7. [Ksql Terminal] Describe the stream:
   
   ```sql
   DESCRIBE impressions;
   DESCRIBE impressions2;
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-22-54-54-image.png)

# Confluent Kafka: Custom Schema Creation

## Working with AVRO Schema

### 1. AVRO Schema Structure

```json
{
    "namespace": "<Unique identifying namespace>",
    "name": "<Name of this schema>",
    "type": "record",
    "fields": [...]
}
```

Why use AVRO schema?

AVRO schema provides:

1. Strong data typing

2. Schema evolution capabilities

3. Compact serialization format

4. Built-in documentation through schema definitions

### 2. Creating Running ID Schema

2. [Confluent Terminal] Construct a simple schema “runningid.avro” that generates running id.
   Test out with ksql datagen using this new schema

```json
{
    "namespace": "datagen",
    "name": "runningid",
    "type": "record",
    "fields": [
        {
            "name": "id",
            "type": {
                "type": "long",
                "arg.properties": {
                    "iteration": {
                        "start": 0
                    }
                }
            }
        }
    ]
}
```

What is the purpose of arg.properties?

arg.properties allows you to:

1. Define generation rules for fields

2. Set starting values and increments

3. Control how data is generated for testing

### 3. Creating Choices Schema

1. [Confluent Terminal] Construct a simple schema “choicesinlife.avro” that generates random
   data based on fixes choices. Test out with ksql datagen using this new schema
   
   ```json
   {
    "namespace": "datagen",
    "name": "choicesinlife",
    "type": "record",
    "fields": [
        {
            "name": "choice",
            "type": "int"
        },
        {
            "name": "description",
            "type": "string"
        }
    ],
    "arg.properties": {
        "options": [
            {
                "choice": 1,
                "description": "Happy"
            },
            {
                "choice": 2,
                "description": "Sad"
            }
        ]
    }
   }
   ```

How do options work in the schema?

The options array:

1. Defines preset values for data generation

2. Creates mappings between different fields

3. Ensures consistent relationships in generated data

# Testing Custom Schemas in Confluent Kafka

## Testing Running ID Schema

```bash
ksql-datagen schema=~/runningid.avro format=json topic=running_ids key=id
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-00-54-image.png)

```sql
# Ksql Terminal
CREATE STREAM running_ids_stream (
    id BIGINT
) WITH (
    KAFKA_TOPIC='running_ids',
    VALUE_FORMAT='JSON'
);

SELECT * FROM running_ids_stream EMIT CHANGES LIMIT 5;
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-00-06-image.png)

## Testing Choices Schema

```bash
# Confluent Terminal
ksql-datagen schema=~/choicesinlife.avro format=json topic=life_choices key=choice
```

```sql
# Ksql Terminal
CREATE STREAM life_choices_stream (
    choice INTEGER,
    description VARCHAR
) WITH (
    KAFKA_TOPIC='life_choices',
    VALUE_FORMAT='JSON'
);

SELECT * FROM life_choices_stream EMIT CHANGES LIMIT 5;
```

How to stop the data generation?

1. Use Ctrl+C in the Confluent Terminal where datagen is running
2. The generation will stop gracefully
3. The created streams will persist until explicitly dropped

# Confluent Kafka: Advanced KSQL Queries

## A. Basic Filtering

### 1. Using LIMIT

```sql
SELECT pageid 
FROM pageviews_original
LIMIT 3;
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-01-32-image.png)

Why use LIMIT in streaming queries?

LIMIT is useful for:

1. Testing query structure without overwhelming output

2. Debugging data formats

3. Initial data exploration

## B. Stream Joins

### 1. Creating Enriched Pageviews

```sql
CREATE STREAM pageviews_enriched AS
SELECT users_original.userid AS userid, 
       pageid, 
       regionid, 
       gender
FROM pageviews_original
LEFT JOIN users_original
ON pageviews_original.userid = users_original.userid;
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-01-54-image.png)

Why use LEFT JOIN instead of INNER JOIN?

LEFT JOIN is preferred because:

1. It retains all pageview records
2. Handles cases where user data might be missing
3. Prevents data loss from the main event stream

4. [Confluent Terminal] Start the streams.
   
   ```bash
   # Confluent Terminal
   ksql-datagen quickstart=pageviews topic=pageviews
   ```

# Start second confluent terminal

ksql-datagen quickstart=users topic='users_kafka_topic_json'

```
4. [ksql Terminal] Inspect the newly created stream
```sql
SELECT * FROM pageviews_enriched emit changes;
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-03-41-image.png)[Confluent Terminal] Stop the streams.

## C. Advanced Filtering

### 1. Gender-Based Filtering

5. [ksql Terminal] Create a new persistent query using a “where” to limit the content
   
   ```sql
   CREATE STREAM pageviews_female AS
   SELECT * FROM pageviews_enriched
   WHERE gender = 'FEMALE';
   ```

6. [ksql Terminal] Create a new persistent query using a “like” and at the same time, output the
   query to a kafka topic
   
   ```sql
   CREATE STREAM pageviews_female_like_89
   AS SELECT * FROM pageviews_female
   WHERE regionid LIKE '%_8' OR regionid LIKE '%_9';
   ```
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-04-25-image.png)
   
   7. [Confluent Terminal] Start the streams.
   
   ```bash
   # Confluent Terminal
   ksql-datagen quickstart=pageviews topic=pageviews
   ```

# Another confluent terminal

ksql-datagen quickstart=users topic='users_kafka_topic_json'

actually the user datagen is optional

```
```sql
select * from pageviews_female_like_89 emit changes;
list topics;
list streams;
```

4. [Confluent Terminal] Stop the streams.

### 2. Windowed Aggregations

7. [ksql Terminal] Create a new persistent query that counts the pageviews for each region and
   gender combination in a tumbling window of 30 seconds when the count is greater than 1
   
   ```sql
   CREATE TABLE pageviews_regions
   WITH (VALUE_FORMAT='avro') AS 
   SELECT gender + '|+|' + regionid, 
       COUNT(*) as numusers
   FROM pageviews_enriched
   WINDOW TUMBLING (size 30 second)
   GROUP BY gender + '|+|' + regionid
   HAVING COUNT(*) > 1;
   ```
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-07-58-image.png)
   
   8. [Confluent Terminal] Start the streams.
   
   ```bash
   # Confluent Terminal
   ksql-datagen quickstart=pageviews topic=pageviews
   ```

# Another confluent terminal

ksql-datagen quickstart=users topic='users_kafka_topic_json'

```
```sql
SELECT * FROM pageviews_regions emit changes;
```

4. [Confluent Terminal] Stop the streams.

What is a TUMBLING WINDOW?

A tumbling window:

1. Groups records into fixed time intervals

2. Windows don't overlap

3. Each record belongs to exactly one window

4. Useful for time-based aggregations

## D. Query Inspection

### 1. View Active Queries

```sql
SHOW QUERIES;
DESCRIBE PAGEVIEWS_REGIONS EXTENDED;
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-26-23-08-55-image.png)

ksql> DESCRIBE PAGEVIEWS_REGIONS EXTENDED;

Name                 : PAGEVIEWS_REGIONS
Type                 : TABLE
Timestamp field      : Not set - using <ROWTIME>
Key format           : KAFKA
Value format         : AVRO
Kafka topic          : PAGEVIEWS_REGIONS (partitions: 1, replication: 1)
Statement            : CREATE TABLE PAGEVIEWS_REGIONS WITH (CLEANUP_POLICY='compact,delete', KAFKA_TOPIC='PAGEVIEWS_REGIONS', PARTITIONS=1, REPLICAS=1, RETENTION_MS=604800000, VALUE_FORMAT='avro') AS SELECT
  ((PAGEVIEWS_ENRICHED.GENDER + '|+|') + PAGEVIEWS_ENRICHED.REGIONID) KSQL_COL_0,
  COUNT(*) NUMUSERS
FROM PAGEVIEWS_ENRICHED PAGEVIEWS_ENRICHED
WINDOW TUMBLING ( SIZE 30 SECONDS )
GROUP BY ((PAGEVIEWS_ENRICHED.GENDER + '|+|') + PAGEVIEWS_ENRICHED.REGIONID)
HAVING (COUNT(*) > 1)
EMIT CHANGES;

Field      | Type
---------------------------------------------------------------------

 KSQL_COL_0 | VARCHAR(STRING)  (primary key) (Window type: TUMBLING)
 NUMUSERS   | BIGINT

---------------------------------------------------------------------

Queries that write from this TABLE
-----------------------------------

CTAS_PAGEVIEWS_REGIONS_23 (RUNNING) : CREATE TABLE PAGEVIEWS_REGIONS WITH (CLEANUP_POLICY='compact,delete', KAFKA_TOPIC='PAGEVIEWS_REGIONS', PARTITIONS=1, REPLICAS=1, RETENTION_MS=604800000, VALUE_FORMAT='avro') AS SELECT   ((PAGEVIEWS_ENRICHED.GENDER + '|+|') + PAGEVIEWS_ENRICHED.REGIONID) KSQL_COL_0,   COUNT(*) NUMUSERS FROM PAGEVIEWS_ENRICHED PAGEVIEWS_ENRICHED WINDOW TUMBLING ( SIZE 30 SECONDS )  GROUP BY ((PAGEVIEWS_ENRICHED.GENDER + '|+|') + PAGEVIEWS_ENRICHED.REGIONID) HAVING (COUNT(*) > 1) EMIT CHANGES;

For query topology and execution plan please run: EXPLAIN <QueryId>

Runtime statistics by host
-------------------------

Host               | Metric           | Value      | Last Message
------------------------------------------------------------------------------

 ksqldb-server:8088 | messages-per-sec |          1 | 2024-11-26T15:09:15.41Z
 ksqldb-server:8088 | total-messages   |        142 | 2024-11-26T15:09:15.41Z

------------------------------------------------------------------------------

(Statistics of the local KSQL server interaction with the Kafka topic PAGEVIEWS_REGIONS)

Consumer Groups summary:

Consumer Group       : _confluent-ksql-default_query_CTAS_PAGEVIEWS_REGIONS_23

Kafka topic          : PAGEVIEWS_ENRICHED
Max lag              : 2

Partition | Start Offset | End Offset | Offset | Lag
------------------------------------------------------

0         | 0            | 608        | 606    | 2
------------------------------------------------------

Kafka topic          : _confluent-ksql-default_query_CTAS_PAGEVIEWS_REGIONS_23-Aggregate-GroupBy-repartition
Max lag              : 4

Partition | Start Offset | End Offset | Offset | Lag
------------------------------------------------------

0         | 124          | 172        | 168    | 4
------------------------------------------------------

Why inspect queries?

Query inspection helps:

1. Monitor query performance
2. Debug issues
3. Understand data flow
4. Verify query configurations
