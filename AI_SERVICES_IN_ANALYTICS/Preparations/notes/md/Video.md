# Video Analysis with Google Cloud Vision API

## Understanding Video Label Detection

### Background
The Google Cloud Video Intelligence API provides sophisticated video content analysis through three distinct levels:

1. Frame Level Analysis
   - Processes individual frames (1 frame/second)
   - Identifies objects, scenes, and activities
   - Provides temporal precision

2. Shot Level Analysis
   - Automatically detects scene changes
   - Groups related frames into shots
   - Labels content within each shot

3. Segment Level Analysis
   - User-defined time ranges
   - Custom analysis boundaries
   - Flexible content grouping

## Implementation

### Library Setup
```python
from google.cloud import videointelligence_v1 as videointelligence
import os
import io
```

<details>
<summary>Why use videointelligence_v1?</summary>

- Provides stable API version
- Ensures backward compatibility
- Includes all core video analysis features
- Maintains consistent method signatures
</details>

### Environment Configuration
```python
# Verify service key presence
!dir it3386_practical.json

# Check current credentials
%env GOOGLE_APPLICATION_CREDENTIALS

# Set service key
%env GOOGLE_APPLICATION_CREDENTIALS=it3386-2024-s2.json
```

### Video Analysis Function
```python
def analyze_label(path): 
    # Initialize client
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.LABEL_DETECTION]

    # Configure analysis mode
    mode = videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE
    config = videointelligence.LabelDetectionConfig(label_detection_mode=mode)
    context = videointelligence.VideoContext(label_detection_config=config)

    # Read video content
    with io.open(path, "rb") as movie:
        input_content = movie.read()

    # Submit analysis request
    operation = video_client.annotate_video(
        request={
            "features": features, 
            "input_content": input_content, 
            "video_context": context
        }
    )

    result = operation.result(timeout=300)
    print("\nFinished processing.")

    # Process video/segment level label annotations
    segment_labels = result.annotation_results[0].segment_label_annotations
    for i, segment_label in enumerate(segment_labels):
        print("Video label description: {}".format(segment_label.entity.description))
        for category_entity in segment_label.category_entities:
            print(
                "\tLabel category description: {}".format(category_entity.description)
            )

        for i, segment in enumerate(segment_label.segments):
            start_time = (
                segment.segment.start_time_offset.seconds
                + segment.segment.start_time_offset.microseconds / 1e6
            )
            end_time = (
                segment.segment.end_time_offset.seconds
                + segment.segment.end_time_offset.microseconds / 1e6
            )
            positions = "{}s to {}s".format(start_time, end_time)
            confidence = segment.confidence
            print("\tSegment {}: {}".format(i, positions))
            print("\tConfidence: {}".format(confidence))
        print("\n")

    # Process shot level label annotations
    shot_labels = result.annotation_results[0].shot_label_annotations
    for i, shot_label in enumerate(shot_labels):
        print("Shot label description: {}".format(shot_label.entity.description))
        for category_entity in shot_label.category_entities:
            print(
                "\tLabel category description: {}".format(category_entity.description)
            )

        for i, shot in enumerate(shot_label.segments):
            start_time = (
                shot.segment.start_time_offset.seconds
                + shot.segment.start_time_offset.microseconds / 1e6
            )
            end_time = (
                shot.segment.end_time_offset.seconds
                + shot.segment.end_time_offset.microseconds / 1e6
            )
            positions = "{}s to {}s".format(start_time, end_time)
            confidence = shot.confidence
            print("\tSegment {}: {}".format(i, positions))
            print("\tConfidence: {}".format(confidence))
        print("\n")

    # Process frame level label annotations
    frame_labels = result.annotation_results[0].frame_label_annotations
    for i, frame_label in enumerate(frame_labels):
        print("Frame label description: {}".format(frame_label.entity.description))
        for category_entity in frame_label.category_entities:
            print(
                "\tLabel category description: {}".format(category_entity.description)
            )

        # Each frame_label_annotation has many frames,
        # here we print information only about the first frame.
        frame = frame_label.frames[0]
        time_offset = frame.time_offset.seconds + frame.time_offset.microseconds / 1e6
        print("\tFirst frame time offset: {}s".format(time_offset))
        print("\tFirst frame confidence: {}".format(frame.confidence))
        print("\n")
    
    return result
```

<details>
<summary>Understanding SHOT_AND_FRAME_MODE</summary>

This mode enables:
- Frame-by-frame analysis (temporal precision)
- Shot boundary detection (scene changes)
- Combined results for comprehensive analysis
- Optimal for most video analysis tasks
</details>

<details>
<summary>Time Offset Calculation</summary>

The time calculation:
- Combines seconds and microseconds
- Converts to floating-point seconds
- Provides precise temporal locations
- Enables accurate segment identification
</details>

### Executing Analysis
```python
# Analyze video file
video_filename = 'oh_2021_short.mp4'
result_label = analyze_label(video_filename)

# Extract frame labels
frame_labels = result_label.annotation_results[0].frame_label_annotations

# Examine specific frame
print(frame_labels[11])

# View frame details
print(frame_labels[0].frames)
```

<details>
<summary>Response Structure</summary>

The API response contains:
- frame_label_annotations: Per-frame labels
- shot_label_annotations: Scene-level labels
- segment_label_annotations: Custom segment labels
Each annotation includes:
- Entity description
- Confidence score
- Temporal information
- Category classifications
</details>

<details>
<summary>Frame Labels Format</summary>

Each frame label contains:
- time_offset: Temporal position
- confidence: Detection certainty
- entity: Detected object/action
- category: Classification group
</details>

# Frame Analysis and Data Processing

## Exploring Frame Data

### Frame Label Structure
```python
# Examine first frame's data
frame_labels[0]
```

<details>
<summary>Understanding Frame Label Objects</summary>

Each frame label contains:
- entity: Object or action detected
- category_entities: Classification categories
- frames: List of temporal detections
- confidence scores per detection
</details>

### Data Consolidation
```python
records = []
for label in frame_labels:
    record = {'entity':'', 'category':'', 'frame':'', 'confidence':''}
    list_frames = []
    list_confidence = []
    list_categories = []
    
    # Extract entity description
    record['entity'] = label.entity.description
    
    # Process categories
    for cat_ent in label.category_entities:
        list_categories.append(cat_ent.description)
    record['category'] = ','.join(list_categories)
    
    # Process frames and confidence
    for frame in label.frames:
        list_frames.append(str(frame.time_offset))
        list_confidence.append(str(round(frame.confidence*100)))
    
    '''
    record['entity'] = label.entity.description
    if 'category_entities' in label:
        for category in label.categoryEntities:
            list_categories.append(category.description)
            
    record['category'] = ','.join(list_categories)
    
    for frame in label.frames:
        list_frames.append(frame.timeOffset)
        list_confidence.append(str(round(frame.confidence*100)))
    '''

    record['frame'] = ','.join(list_frames)
    record['confidence'] = ','.join(list_confidence)
    records.append(record)
```

<details>
<summary>Record Structure Design</summary>

The record dictionary format:
- entity: Main detected object/action
- category: Hierarchical classifications
- frame: Temporal positions
- confidence: Detection certainty percentages
</details>

## Visualization and Analysis

### HTML Table Generation
```python
from IPython.display import HTML, display

html = '<table><thead><tr><th>Entity</th><th>Category</th><th>Frame At</th><th>Confidence (%)</th></tr></thead><tbody>'

for record in records:
    entity = record['entity']
    category = record['category'].replace(',', '<br>')
    frame = record['frame'].replace(',', '<br>')
    confidence = record['confidence'].replace(',', '<br>')
    row = '<tr style="vertical-align:top"><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
        entity, category, frame, confidence)
    html = html + row

html = html + '</tbody></table>'
display(HTML(html))
```

<details>
<summary>Table Formatting Choices</summary>

- vertical-align:top: Aligns content to top for readability
- <br> replacements: Converts commas to line breaks for multiple values
- HTML structure: Provides organized view of detection results
</details>

## Shot Analysis

### Examining Shot Labels
```python
# Get shot annotations
label_shots = result_label.annotation_results[0].shot_label_annotations

# Examine first shot
label_shots[0]

# Find multi-segment label
label_shots[4]
```

<details>
<summary>Shot vs Frame Labels</summary>

Shot labels differ from frame labels:
- Cover longer time segments
- Identify scene-level content
- Track continuous actions/objects
- Provide broader context
</details>

### Processing Shot Data
```python
for idx, label in enumerate(label_shots):
    print('{} Entity: {}'.format(idx, label.entity.description))
    
    for category in label.category_entities:
        print('Category: {}'.format(category.description))
    
    for segment in label.segments:
        print('Start Time Offset: {}'.format(segment.segment.start_time_offset))
        print('End Time Offset: {}'.format(segment.segment.end_time_offset))
        print('Confidence: {}'.format(segment.confidence))
    
    print('\n')
```

<details>
<summary>Shot Segment Structure</summary>

Each shot segment contains:
- start_time_offset: Beginning of scene
- end_time_offset: End of scene
- confidence: Detection certainty
- entity description: Main content label
</details>

# Video Frame Extraction and Analysis

## Setup Output Directory
```python
# Create directory for extracted frames
dir_output = 'output'
if not os.path.isdir(dir_output):
    os.makedirs(dir_output)
```

<details>
<summary>Directory Structure Purpose</summary>

The output directory:
- Organizes extracted frames
- Stores generated GIFs and MP4s
- Maintains clean workspace
- Enables systematic analysis
</details>

## Video Processing Implementation

### MoviePy Setup
```python
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
```

### Frame Extraction Function
```python
def get_gif_and_mp4_filename(video_source_filename, start_time, stop_time, entity):
    # Generate filenames
    output_filename_video = str(Path(dir_output + '/' + entity.replace(' ', '_') + '.mp4'))
    output_filename_gif = str(Path(dir_output + '/' + entity.replace(' ', '_') + '.gif'))

    # Extract video segment
    clip = VideoFileClip(video_source_filename, audio=False).subclip(start_time, stop_time)
    clip.write_videofile(output_filename_video)
    clip.close()
    
    # Create GIF
    clip_segment = VideoFileClip(output_filename_video, audio=False)
    clip_segment.write_gif(output_filename_gif)
    return output_filename_gif, output_filename_video
```

<details>
<summary>Function Parameters Explained</summary>

- video_source_filename: Source video path
- start_time: Beginning of segment
- stop_time: End of segment
- entity: Label for output files
- Returns both GIF and MP4 paths for flexibility
</details>

## Analyzing Specific Segments

### Extract Segment Example
```python
# Select specific shot index
index = 8
print(label_shots[index])
time_offset_start = label_shots[index].segments[0].segment.start_time_offset.seconds
time_offset_end = label_shots[index].segments[0].segment.end_time_offset.seconds

entity_description = label_shots[index].entity.description
print(entity_description)

# Generate video files
filename_gif, filename_mp4 = get_gif_and_mp4_filename(
    video_filename, 
    time_offset_start, 
    time_offset_end, 
    entity_description
)
```
<details>
<summary>Example entity</summary>

```text
Entity: lecture
Start Time Offset: 22.720s
End Time Offset: 23.760s
```

</details>

### Display Results
```python
from IPython import display
display.Image(filename_gif, width=320)
```

## Frame-by-Frame Analysis

### Extract Individual Frames
```python
clip_segment = VideoFileClip(filename_mp4)
clip_segment.write_images_sequence('output/frame%04d.jpg', verbose=True)
```

<details>
<summary>Frame Extraction Format</summary>

- %04d: 4-digit zero-padded frame numbers
- .jpg format for universal compatibility
- Sequential naming for easy processing
- Verbose output for progress tracking
</details>

### Display First Frame
```python
print(entity_description)
display.Image('output/frame0000.jpg', width=320)
```

<details>
<summary>Analysis Workflow</summary>

The complete process:
1. Extract video segment based on detection
2. Convert to both MP4 and GIF formats
3. Extract individual frames
4. Display results for verification
5. Compare with AI predictions
</details>
