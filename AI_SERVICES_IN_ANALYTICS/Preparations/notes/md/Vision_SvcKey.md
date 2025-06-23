# Google Cloud Vision API Authentication and Setup Guide

## Background
The Google Cloud Vision API provides powerful image analysis capabilities through a simple client library interface. This guide covers authentication setup and implementation of three key features: face detection, landmark detection, and document text detection.

## Prerequisites Setup

### Required Files
```plaintext
- it3386-2024-s2.json (service key)
- Image files from week 1 practical
- oh_2021_short.mp4
- Anaconda environment with vision libraries
```

<details>
<summary>Why these prerequisites?</summary>

- Service key enables secure API authentication
- Sample images allow immediate testing
- Anaconda environment ensures consistent library versions
- Video file supports extended functionality testing
</details>

## Authentication Configuration

### 1. Verify Service Key
```python
!dir *.json
```

### 2. Check Current Credentials
```python
%env GOOGLE_APPLICATION_CREDENTIALS
```

### 3. Set Service Key Path
```python
%env GOOGLE_APPLICATION_CREDENTIALS=it3386-2024-s2.json
```

<details>
<summary>Authentication Process Explained</summary>

1. Check JSON files to confirm key presence
2. Verify current credential state
3. Set credentials to new service key
4. Environment variable enables automatic authentication
</details>

## Core Implementation

### 1. Face Detection
```python
# Import required libraries
from google.cloud import vision
import io

# Face detection function
def detect_faces(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                      'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    
    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])
        print('face bounds: {}'.format(','.join(vertices)))
    return response
```

<details>
<summary>Face Detection Components</summary>

- ImageAnnotatorClient: Creates API connection
- face_detection: Analyzes facial features
- face_annotations: Contains detection results
- likelihood_name: Maps numeric values to readable states
</details>

### 2. Landmark Detection
```python
def detect_landmarks(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    
    for landmark in landmarks:
        print(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print('Latitude {}'.format(lat_lng.latitude))
            print('Longitude {}'.format(lat_lng.longitude))
    return response
```

<details>
<summary>Landmark Detection Features</summary>

- Identifies famous locations and buildings
- Provides geographic coordinates
- Returns landmark names and descriptions
- Includes confidence scores
</details>

### 3. Document Text Detection
```python
def detect_document(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))
    return response
```

<details>
<summary>Document Detection Structure</summary>

Hierarchical text processing:
1. Pages: Document containers
2. Blocks: Text sections
3. Paragraphs: Text groups
4. Words: Individual terms
5. Symbols: Characters
</details>

## Function Execution

### Face Detection
```python
response = detect_faces('nyp_cafe.jpg')
print(response)
```

### Landmark Detection
```python
response = detect_landmarks('place.jpg')
print(response)
```

### Document Detection
```python
response = detect_document('note.jpg')
print(response)
```

<details>
<summary>Response Handling</summary>

Each response includes:
- Detection results
- Confidence scores
- Error messages (if any)
- Structured data for further processing
</details>
