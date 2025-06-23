# Part 1: Time Windows and Emitting Changes in Confluent Kafka

### A.1. Generating Data Aggregation via Time Window

1. Open 2 Terminals (Label them as Confluent Terminal and Ksql Terminal)

2. Ensure the following pre-requisites are met:
   
   a. [Ksql Terminal] KsqlDB Server is up and running
   
   ```bash
   confluent local services status
   confluent local services ksql-server start
   ```
   
   b. [Confluent Terminal] Ensure the Confluent Home and Path environment variables are set up
   
   ```sh
   echo $PATH
   echo $CONFLUENT_HOME
   ```

3. [Confluent Terminal] Generate dummy stream using the built-in schema
   
   ```sh
   ksql-datagen quickstart=orders topic=orders_topic iterations=100 msgRate=10
   ```
   
   | Parameters | Description                                                 |
   | ---------- | ----------------------------------------------------------- |
   | Iterations | The maximum number of records to generate                   |
   | msgRate    | The rate to produce the message at, in messages per seconds |

4. [Ksql Terminal] List the Topic followed by create a stream
   
   ```sql
   LIST TOPICS;
   PRINT 'orders_topic';
   
   # Run the ksql-datagen quickstart=orders topic=orders_topic
   command to see the real time streaming at the ksql terminal
   ```
   
   What do the datagen parameters mean?
   
   - iterations=100: Generates exactly 100 records
   
   - msgRate=10: Produces 10 messages per second
   
   - quickstart=orders: Uses the pre-defined orders schema template

5. [Ksql Terminal] Create the Order Raw stream if it is not created already
   
   ```sql
   CREATE STREAM orders_raw (
     ordered INT KEY,
     itemid VARCHAR,
     address STRUCT<
       city VARCHAR,
       state VARCHAR,
       zip INT
     >,
     ordertime TIMESTAMP
   )
   WITH (
     KAFKA_TOPIC='orders_topic',
     VALUE_FORMAT='JSON'
   );
   ```
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-33-38-image.png)
   
   Why do we create a structured stream?
- ## Provides a schema-based view of the Kafka topic
- Enables SQL-like queries on streaming data
- Allows for proper data type handling and validation
6. [Ksql Terminal] Perform Fixed Window Aggregation on orders_raw
   
   ```sql
   SELECT itemid,
   from_unixtime(WINDOWSTART) as Window_Start,
   from_unixtime(WINDOWEND) as Window_End,
   from_unixtime(max(ROWTIME)) as Window_Emit,
   count(itemid) as number_of_orders
   FROM orders_raw
   WINDOW TUMBLING (SIZE 60 days, GRACE PERIOD 0 minute)
   GROUP BY itemid
   EMIT CHANGES;
   ```
   
   Question : What did you observe in the aggregated data’s time Window range? Is it
   expected? (no answer was provided in practical)
   
   ![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-34-32-image.png)

## A.2. Specify the use of Event Time

## A.2. Specify the use of Event Time

1. [Ksql Terminal] Create a new Stream called "Orders_Raw_ts"
   
   ```sql
   CREATE STREAM orders_raw_ts (
     ordered INT KEY,
     itemid VARCHAR,
     address STRUCT<
       city VARCHAR,
       state VARCHAR,
       zip INT
     >,
     ordertime TIMESTAMP
   )
   WITH (
     KAFKA_TOPIC='orders_topic',
     VALUE_FORMAT='JSON',
     TIMESTAMP='ordertime'
   );
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-35-57-image.png)

What's the significance of timestamp = 'ordertime'?

- Tells KSQL to use the event's actual timestamp (ordertime) instead of processing time
- Ensures window calculations are based on when events occurred rather than when they were processed
- Critical for accurate time-based analytics when events might arrive out of order
2. [Ksql Terminal] Run the following select statement
   
   ```sql
   SELECT itemid,
   from_unixtime(WINDOWSTART) as Window_Start,
   from_unixtime(WINDOWEND) as Window_End,
   from_unixtime(max(ROWTIME)) as Window_Emit,
   count(itemid) as number_of_orders
   FROM orders_raw_ts
   WINDOW TUMBLING (SIZE 60 days, GRACE PERIOD 0 minute)
   GROUP BY itemid
   EMIT CHANGES;
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-42-01-image.png)

What are the window parameters?

- SIZE 60 days: Defines the duration of each tumbling window
- GRACE PERIOD 0 minute: Time allowed for late-arriving events
- EMIT CHANGES: Continuously emits results as new data arrives

### 3. Experiment with Window Parameters

Try these modifications:

```sql
-- Example with different window size
WINDOW TUMBLING (SIZE 30 days, GRACE PERIOD 0 minute)

-- Example with grace period
WINDOW TUMBLING (SIZE 60 days, GRACE PERIOD 5 minute)

-- Example with sliding window
WINDOW HOPPING (SIZE 60 days, ADVANCE BY 30 days)
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-38-20-image.png)

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-39-32-image.png)

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-40-34-image.png)

How do different window settings affect results?

- Larger windows aggregate more data but have delayed final results
- Grace periods allow for late data but increase result latency
- Sliding windows create overlapping time periods, useful for trending analysis

## B. Distinguish Data Aggregation between Emit Final and Emit Changes

### B.1. Compare EMIT FINAL vs EMIT CHANGES Behavior

1. Run the EMIT CHANGES version first:
   
   ```sql
   -- [Ksql Terminal]
   SELECT itemid,
    from_unixtime(WINDOWSTART) as Window_Start,
    from_unixtime(WINDOWEND) as window_End,
    from_unixtime(max(ROWTIME)) as Window_Emit,
    count(itemid) as number_of_orders
   FROM orders_raw_ts
    WINDOW TUMBLING (SIZE 60 days, GRACE PERIOD 0 minute)
    GROUP BY itemid
    EMIT CHANGES;
   ```

2. Then run the EMIT FINAL version:
   
   ```sql
   SELECT itemid,
    from_unixtime(WINDOWSTART) as Window_Start,
    from_unixtime(WINDOWEND) as window_End,
    from_unixtime(max(ROWTIME)) as Window_Emit,
    count(itemid) as number_of_orders
   FROM orders_raw_ts
    WINDOW TUMBLING (SIZE 60 days, GRACE PERIOD 0 minute)
    GROUP BY itemid
    EMIT FINAL;
   ```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-45-10-image.png)

What's the difference between EMIT CHANGES and EMIT FINAL?

- EMIT CHANGES:
  
  - Shows intermediate results as data arrives
  - Updates results continuously
  - Useful for real-time monitoring

- EMIT FINAL:
  
  - Only shows results when window closes
  - Provides one final result per window
  - Better for historical analysis

### B.2. Experimentation Tasks

Try these modifications:

```sql
-- Example with longer window
WINDOW TUMBLING (SIZE 90 days, GRACE PERIOD 0 minute)

-- Example with extended grace period
WINDOW TUMBLING (SIZE 60 days, GRACE PERIOD 10 minute)
```

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-46-08-image.png)

![](/Users/siewweiheng/Library/Application%20Support/marktext/images/2024-11-27-08-47-01-image.png)

How do window modifications affect the results?

1. First Query (30-day Tumbling Window):
- Shows orders grouped in non-overlapping 30-day windows

- Each item appears only once in its respective window

- The number_of_orders column shows 1 for each item, indicating single purchases
2. Second Query (60-day Tumbling Window with 5-minute Grace Period):
- Uses larger 60-day windows

- Grace period allows late-arriving data within 5 minutes after window closes

- Shows more items per window due to longer duration
3. Third Query (60-day Hopping Window with 30-day advance):
- Creates overlapping windows (60-day duration, moving forward 30 days each time)

- Some items appear in multiple windows due to overlap

- Example: Item_749 appears in two consecutive windows
4. Fourth & Fifth Queries (60-day Tumbling Window with EMIT CHANGES/FINAL):
- EMIT CHANGES: Shows results as they arrive

- EMIT FINAL: Shows only final results after window closes

- Both show non-overlapping 60-day periods
5. Last Query (60-day Tumbling Window with 10-minute Grace Period):
- Similar to second query but with longer grace period

- Shows how items are distributed across different time windows
