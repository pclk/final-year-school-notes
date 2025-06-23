# Data Center Temperature Monitoring System

## Objective
Simulate incoming temperature data from data center sensors, process it using Dataflow, and visualize the results in BigQuery and Looker Studio.

## Data Source
- DC_Temp_data.csv.gz

## Setup Instructions

### 1. Prepare the Environment (SSH 1)
1. Access your Compute Engine VM:
   - Navigate to GCP Console > Compute Engine > VM instances
   - Click SSH button next to your VM instance

2. Clone repository and set environment:
```bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
source /training/project_env.sh
```

### 2. Set up BigQuery Dataset
1. Navigate to BigQuery in GCP Console
2. Click on project ID
3. Create dataset named "demos"

### 3. Upload Data to Bucket
Upload the data file to your GCS bucket

### 4. Prepare Data Publisher Script (SSH 1)
1. Configure sensor script:
```bash
sudo vim /training/sensor_magic.sh
gsutil cp gs://qwiklabs-gcp-00-3e2cd7b6f26e/DC_Temp_Data.csv.gz ~/training-data-analyst/courses/streaming/publish/
```

- The `gsutil cp` command copies the temperature data file (`DC_Temp_Data.csv.gz`) from a Google Cloud Storage bucket to the local working directory
- This compressed CSV file contains the historical/sample temperature data that will be used to simulate real-time sensor readings
- The data is copied to the `publish` directory where the publisher script (`send_sensor_data.py`) will access it to stream data to Pub/Sub


```py
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
TOPIC = "sandeigo"
INPUT = "DC_Temp_Data.csv.gz"
```

2. Navigate to publisher directory:
```bash
cd training-data-analyst/courses/streaming/publish/
sudo vim send_sensor_data.py
```

### 5. Simulate Streaming Data with Pub/Sub
```bash
gcloud services enable pubsub.googleapis.com
/training/sensor_magic.sh
```

### 6. Set up Dataflow
1. Set environment variables:
```bash
source /training/project_env.sh
cd ~/training-data-analyst/courses/streaming/process/sandiego
gcloud services disable dataflow.googleapis.com --force
gcloud services enable dataflow.googleapis.com
```

2. Configure Dataflow Schema:
```js
// BigQuery table schema
fields.add(new TableFieldSchema().setName("timestamp").setType("TIMESTAMP"));
fields.add(new TableFieldSchema().setName("avg_temp").setType("FLOAT"));
fields.add(new TableFieldSchema().setName("location").setType("STRING"));
// CSV line format
String line = Instant.now().toString() + "," + speed + "," + stationKey;
LaneInfo info = LaneInfo.newLaneInfo(line);
// BigQuery row mapping
row.set("timestamp", info.getTimestamp());
row.set("avg_temp", info.getSpeed());
row.set("location", info.getSensorKey());
```

Final AverageSpeeds file:
```java
package com.google.cloud.training.dataanalyst.sandiego;

import java.util.ArrayList;
import java.util.List;

import org.apache.beam.runners.dataflow.options.DataflowPipelineOptions;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.gcp.pubsub.PubsubIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.Mean;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.windowing.SlidingWindows;
import org.apache.beam.sdk.transforms.windowing.Window;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.joda.time.Duration;
import org.joda.time.Instant;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;

/**
 * A dataflow pipeline that computes average speeds in each lane
 * 
 * @author vlakshmanan
 *
 */
public class AverageSpeeds {

  public static interface MyOptions extends DataflowPipelineOptions {
    @Description("Over how long a time period should we average? (in minutes)")
    @Default.Double(5.0)
    Double getAveragingInterval();

    void setAveragingInterval(Double d);

    @Description("Simulation speedup factor. Use 1.0 if no speedup")
    @Default.Double(60.0)
    Double getSpeedupFactor();

    void setSpeedupFactor(Double d);
  }

  @SuppressWarnings("serial")
  public static void main(String[] args) {
    MyOptions options = PipelineOptionsFactory.fromArgs(args).withValidation().as(MyOptions.class);
    options.setStreaming(true);
    Pipeline p = Pipeline.create(options);

    String topic = "projects/" + options.getProject() + "/topics/sandiego";
    String avgSpeedTable = options.getProject() + ":demos.average_temperature";

    // if we need to average over 60 minutes and speedup is 30x
    // then we need to average over 2 minutes of sped-up stream
    Duration averagingInterval = Duration
        .millis(Math.round(1000 * 60 * (options.getAveragingInterval() / options.getSpeedupFactor())));
    Duration averagingFrequency = averagingInterval.dividedBy(2); // 2 times
    // in
    // window
    System.out.println("Averaging interval = " + averagingInterval);
    System.out.println("Averaging freq = " + averagingFrequency);

    // Build the table schema for the output table.
    List<TableFieldSchema> fields = new ArrayList<>();
    fields.add(new TableFieldSchema().setName("timestamp").setType("TIMESTAMP"));
    fields.add(new TableFieldSchema().setName("avg_temp").setType("FLOAT"));
    fields.add(new TableFieldSchema().setName("location").setType("STRING"));
    TableSchema schema = new TableSchema().setFields(fields);

    PCollection<LaneInfo> currentConditions = p //
        .apply("GetMessages", PubsubIO.readStrings().fromTopic(topic)) //
        .apply("ExtractData", ParDo.of(new DoFn<String, LaneInfo>() {
          @ProcessElement
          public void processElement(ProcessContext c) throws Exception {
            String line = c.element();
            c.output(LaneInfo.newLaneInfo(line));
          }
        }));

    PCollection<KV<String, Double>> avgSpeed = currentConditions //
        .apply("TimeWindow",
            Window.into(SlidingWindows//
                .of(averagingInterval)//
                .every(averagingFrequency))) //
        .apply("BySensor", ParDo.of(new DoFn<LaneInfo, KV<String, Double>>() {
          @ProcessElement
          public void processElement(ProcessContext c) throws Exception {
            LaneInfo info = c.element();
            String key = info.getSensorKey();
            Double speed = info.getSpeed();
            c.output(KV.of(key, speed));
          }
        })) //
        .apply("AvgBySensor", Mean.perKey());

    avgSpeed.apply("ToBQRow", ParDo.of(new DoFn<KV<String, Double>, TableRow>() {
      @ProcessElement
      public void processElement(ProcessContext c) throws Exception {
        TableRow row = new TableRow();
        String stationKey = c.element().getKey();
        Double speed = c.element().getValue();
        String line = Instant.now().toString() + "," + speed + "," + stationKey; // CSV
        LaneInfo info = LaneInfo.newLaneInfo(line);
        row.set("timestamp", info.getTimestamp());
        row.set("avg_temp", info.getSpeed());
        row.set("location", info.getSensorKey());
        c.output(row);
      }
    })) //
        .apply(BigQueryIO.writeTableRows().to(avgSpeedTable)//
            .withSchema(schema)//
            .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND)
            .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED));

    p.run();
  }
}
```

3. LaneInfo Configuration:
```java
// Field enum
TIMESTAMP, SPEED, SENSORID, LATITUDE, LONGITUDE, FREEWAY_ID, FREEWAY_DIR, LANE;

public String getSensorKey() {
    return get(Field.SENSORID);
}
```

Final LaneInfo file:
```java
package com.google.cloud.training.dataanalyst.sandiego;

import org.apache.beam.sdk.coders.AvroCoder;
import org.apache.beam.sdk.coders.DefaultCoder;

@DefaultCoder(AvroCoder.class)
public class LaneInfo {
  private String[] fields;

  private enum Field {
    TIMESTAMP, SPEED, SENSORID, LATITUDE, LONGITUDE, FREEWAY_ID, FREEWAY_DIR, LANE;
  }

  public LaneInfo() {
    // for Avro
  }

  public static LaneInfo newLaneInfo(String line) {
    String[] pieces = line.split(",");
    LaneInfo info = new LaneInfo();
    info.fields = pieces;
    return info;
  }

  private String get(Field f) {
    return fields[f.ordinal()];
  }

  public String getTimestamp() {
    return fields[Field.TIMESTAMP.ordinal()];
  }

  /**
   * Create unique key for sensor in a particular lane
   * 
   * @return
   */
  public String getSensorKey() {
    return get(Field.SENSORID);
  }

  /**
   * Create unique key for all the sensors for traffic in same direction at a
   * location
   * 
   * @return
   */
  public String getLocationKey() {
    StringBuilder sb = new StringBuilder();
    for (int f = Field.LATITUDE.ordinal(); f <= Field.FREEWAY_DIR.ordinal(); ++f) {
      sb.append(fields[f]);
      sb.append(',');
    }
    return sb.substring(0, sb.length() - 1); // without trailing comma
  }

  public double getLatitude() {
    return Double.parseDouble(get(Field.LATITUDE));
  }

  public double getLongitude() {
    return Double.parseDouble(get(Field.LONGITUDE));
  }

  public String getHighway() {
    return get(Field.FREEWAY_ID);
  }

  public String getDirection() {
    return get(Field.FREEWAY_DIR);
  }

  public int getLane() {
    return Integer.parseInt(get(Field.LANE));
  }

  public double getSpeed() {
    return Double.parseDouble(get(Field.SPEED));
  }
}
```

4. Run Dataflow Job:
```bash
export REGION=us-east1
./run_oncloud.sh $DEVSHELL_PROJECT_ID $BUCKET AverageSpeeds
```

### 7. BigQuery Analysis

1. View Recent Data:
```sql
SELECT *
FROM `[project_id].demos.average_temperature`
ORDER BY timestamp DESC
LIMIT 1000
```

2. Find Latest Timestamp:
```sql
SELECT MAX(timestamp)
FROM `[project_id].demos.average_temperature`
```

3. View data existed 1 minute ago:
```sql
SELECT *
FROM `[project_id].demos.average_temperature`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 MINUTE)
ORDER BY timestamp DESC
LIMIT 100
```

This SQL query is designed to view historical data from exactly 1 minute ago in the BigQuery table. Let's break down its components:

1. `FOR SYSTEM_TIME AS OF` - This is a temporal query clause that allows you to query data as it existed at a specific point in time
   
2. `TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 MINUTE)` - This function:
   - Takes the current timestamp (`CURRENT_TIMESTAMP()`)
   - Subtracts 1 minute from it (`INTERVAL 1 MINUTE`)
   
3. The query will return:
   - All columns (`SELECT *`)
   - From the average_speeds table
   - As they existed exactly 1 minute ago
   - Ordered by timestamp in descending order (newest first)
   - Limited to 100 records

This query is useful for:
- Debugging recent data issues
- Monitoring recent temperature readings
- Verifying data is being properly ingested
- Analyzing temperature patterns with a small delay to ensure data completeness


3. Calculate Temperature Change Rate:
```sql
SELECT
  timestamp,
  location,
  avg_temp,
  avg_temp - COALESCE(LAG(avg_temp, 1) OVER (PARTITION BY location ORDER BY timestamp), avg_temp) AS temp_change_rate
FROM `[project_id].demos.average_temperature`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 MINUTE)
ORDER BY timestamp DESC
LIMIT 100
```


1. The main purpose is to calculate how much the temperature has changed between consecutive readings for each location. Here's how it works step by step:

2. The key part is this calculation:
```sql
avg_temp - COALESCE(LAG(avg_temp, 1) OVER (PARTITION BY location ORDER BY timestamp), avg_temp) AS temp_change_rate
```

Let's break it down:

a) `LAG(avg_temp, 1)` - This window function looks at the previous row's temperature
   - Gets the temperature reading from 1 row before the current row
   
b) `OVER (PARTITION BY location ORDER BY timestamp)` 
   - `PARTITION BY location` - Groups the data by sensor location
   - `ORDER BY timestamp` - Orders readings chronologically
   - This ensures we're comparing temperatures from the same sensor location

c) `COALESCE(..., avg_temp)`
   - If there is no previous reading (first row for a location), use the current temperature
   - This prevents NULL values

d) `avg_temp - ...` 
   - Subtracts the previous temperature from the current temperature
   - Result shows how much temperature changed:
     - Positive number = temperature increased
     - Negative number = temperature decreased

Example:
```
Time       Location    Temp    Change
9:00       1           22      0      (first reading, no change)
9:01       1           24      +2     (increased by 2)
9:02       1           23      -1     (decreased by 1)
```

The query also:
- Uses `FOR SYSTEM_TIME AS OF` to look at data from 1 minute ago
- Orders results by newest first
- Limits to 100 records

This is useful for:
- Detecting sudden temperature spikes
- Monitoring cooling system performance
- Identifying potential equipment issues
- Understanding temperature patterns over time

4. Calculate Time Above Threshold:
```sql
WITH TempWithFlag AS (
  SELECT
    timestamp,
    location,
    avg_temp,
    CASE
      WHEN avg_temp > 24 THEN 1
      ELSE 0
    END AS above_threshold_flag
  FROM `[project_id].demos.average_temperature`
)
SELECT
  location,
  SUM(above_threshold_flag) AS time_above_threshold
FROM TempWithFlag
GROUP BY location
```

ihe query uses a Common Table Expression (CTE) that is identified by the WITH clause and works in two steps:

Step 1: `TempWithFlag` CTE creates a temporary result with a flag:
```sql
WITH TempWithFlag AS (
  SELECT
    timestamp,
    location,
    avg_temp,
    CASE
      WHEN avg_temp > 24 THEN 1  -- If temperature > 24°C, flag = 1
      ELSE 0                     -- If temperature ≤ 24°C, flag = 0
    END AS above_threshold_flag
  FROM `[project_id].demos.average_temperature`
)
```

Step 2: Summarizes the flags by location:
```sql
SELECT
  location,
  SUM(above_threshold_flag) AS time_above_threshold  -- Counts how many times temp was > 24°C
FROM TempWithFlag
GROUP BY location
```

Example of how it works:
```
Original Data:
Time    Location    Temp
9:00    1           25    -> Flag = 1 (above 24)
9:01    1           23    -> Flag = 0 (below 24)
9:02    1           26    -> Flag = 1 (above 24)
9:00    2           22    -> Flag = 0 (below 24)
9:01    2           21    -> Flag = 0 (below 24)

Result:
Location    time_above_threshold
1           2  (was above 24°C twice)
2           0  (never above 24°C)
```

This query is useful for:
- Identifying locations with frequent overheating
- Monitoring cooling system effectiveness
- Compliance with temperature thresholds
- Prioritizing maintenance based on problem areas

The threshold (24°C) can be adjusted based on your data center's requirements.

### 9. Looker Studio Visualization
```sql
SELECT
  MAX(avg_temp) AS maxtemp,
  MIN(avg_temp) AS mintemp,
  AVG(avg_temp) AS avgtemp,
  location
FROM `[project-id].demos.average_temperature`
GROUP BY location
```

1. Pie Chart: "Number of rows per location"

Purpose: This pie chart visualizes the distribution of data across three different locations, "1", "2", and "3". 
Data: The pie chart shows an equal distribution of data across the three locations, with each location accounting for 33.3% of the total rows. This suggests that the data collection streaming in is evenly balanced across all locations.

2. Bar Chart:

Purpose: This bar chart compares the avgtemp, mintemp, and maxtemp values across the three locations.

Data:
The x-axis represents the three locations: "1", "3", and "2" (in that order).
The y-axis represents the temperature, ranging from 0 to 25.
For each location, there are three bars:
Blue: avgtemp
Turquoise: mintemp
Magenta: maxtemp


3. Line Chart:
Purpose: This line chart visualizes the trend of average temperature value over time. 

Data:
The x-axis represents time, with dates progressing from what appears to be "Dec 18, 202...". The time granularity is in minutes.
The y-axis represents the temperature value, ranging from 0 to 25.
Three lines are plotted, corresponding to locations "1" (magenta), "3" (blue), and "2" (turquoise).

Insights:

The analysis shows remarkable consistency and stability in temperatures across all three monitored locations. This is evidenced by the bar chart displaying similar average, minimum, and maximum temperatures, along with flat trend lines in the time series visualization. This consistency could be attributed to several factors: the locations may be in close proximity experiencing similar environmental conditions, the data center's temperature control systems may be functioning effectively, or the analysis period might be too brief to capture significant variations.

The data collection system demonstrates excellent reliability and balance, with each location contributing exactly one-third of the total data points. This equal distribution suggests a robust streaming pipeline where all sensors are functioning optimally with no data loss.

To enhance the current system, implementing anomaly detection would be valuable. This could be achieved through various methods: establishing statistical thresholds based on historical data patterns, employing time-series analysis techniques like moving averages and exponential smoothing, or developing machine learning models specifically trained for anomaly detection. These mechanisms would help identify unusual temperature patterns that might indicate equipment malfunctions, environmental issues, or potential security concerns.

Furthermore, expanding the system to include advanced time-series analysis capabilities would provide deeper insights. This could involve analyzing seasonal patterns and long-term trends in temperature variations throughout the year. Such analysis would be particularly valuable for identifying recurring patterns, such as seasonal temperature fluctuations, and long-term trends that might indicate gradual changes in the data center's thermal characteristics. Additionally, implementing forecasting models would support proactive capacity planning and resource allocation, enabling better preparation for future temperature management challenges.
