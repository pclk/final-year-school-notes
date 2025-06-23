# Google Cloud Vision API Practical Guide

<details>
<summary>Background</summary>

Google Cloud Vision API is a powerful machine learning tool that enables image analysis through pre-trained models. It can detect faces, emotions, landmarks, and objects, making it invaluable for applications ranging from content moderation to user experience enhancement. The API processes images and returns detailed information about visual content, including facial features, emotional expressions, and spatial coordinates.
</details>

## Required Dependencies
```python
import requests 
import base64
import json
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
%matplotlib inline
```

<details>
<summary>Dependency Explanations</summary>

- **PIL and BytesIO**: PIL (Python Imaging Library) is used for image processing tasks like opening, manipulating, and saving images. BytesIO provides a way to work with binary image data in memory, which is essential when processing images without saving intermediate files.
- **%matplotlib inline**: This magic command tells Jupyter notebooks to display plots directly in the notebook rather than in a separate window, making it easier to view results immediately.
</details>

## API Configuration
```python
googleAPIKey = "AIzaSyBhNkvLI0d7ZcHxP95h7slBOs-5OT8nLa0"
googleurl = "https://vision.googleapis.com/v1/images:annotate?key=" + googleAPIKey
req_headers = {'Content-Type': 'application/json'}
```

<details>
<summary>Configuration Notes</summary>

- The 'Content-Type' header tells the API server that we're sending JSON-formatted data. Without it, the server might not process our request correctly or might reject it entirely.
</details>

## Helper Functions
```python
def get_base64(image_filename):
    with open(image_filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string
```

<details>
<summary>Function Details</summary>

- **Base64 Encoding**: Converts binary image data into ASCII text format. This is necessary because:
  - JSON can only contain text data, not binary data
  - It ensures safe transmission across different systems and networks
  - It prevents data corruption during API requests
- **"rb" Mode**: The "rb" flag opens the file in binary read mode, which is necessary for image files to prevent any text encoding/decoding issues.
</details>

## Image Loading and Display
```python
img_filename = 'data/nyp_cafe.jpg'
plt.imshow(plt.imread(img_filename))
plt.axis('off');
```

<details>
<summary>Display Function Details</summary>

- **plt.imshow() benefits:**
  - Better display control
  - Ability to manipulate the image display (like removing axes)
  - Consistent display across different platforms
- **imread() vs PIL.Image.open()**: plt.imread() returns a NumPy array suitable for matplotlib, while PIL.Image.open() returns a PIL Image object better suited for image processing tasks.
</details>

## Face Detection Request
```python
data = {
  'requests': [
    {
      'image': { 'content': get_base64(img_filename).decode('utf-8') },
      'features': [{ 'type': 'FACE_DETECTION' }]
    }
  ]
}

r = requests.post(url=googleurl, headers=req_headers, json=data)

if r.status_code == 200:
  result = r.json()
  print(result)
else:
  print('Error with status')
  print(r.content)
```

<details>
<summary>Request Structure Details</summary>

- **Nested 'requests' array**: Supports batch processing of multiple images in a single request
- **Available feature types:**
  - LABEL_DETECTION for general image labeling
  - LANDMARK_DETECTION for identifying landmarks
  - TEXT_DETECTION for OCR
  - OBJECT_DETECTION for identifying objects
  - SAFE_SEARCH_DETECTION for content moderation
</details>

## Response Analysis

### Pretty Print Response
```python
print(json.dumps(result, indent=4))
```

<details>
<summary>JSON Formatting Details</summary>

- **json.dumps() advantages:**
  - Formatted output with proper indentation
  - String representation of JSON data
  - Better readability for nested structures
  - Control over output formatting through parameters
- **indent=4**: Creates a hierarchical view with standard 4-space indentation for better readability
</details>

### Detailed Feature Analysis
```python
for annotation in annotations:
    print('\n\nEach face')
    print('\n*** Confidence ***')
    print('Detection Confidence: %.2f' % (annotation['detectionConfidence'] * 100))
    print('Landmarking Confidence: %.2f' % (annotation['landmarkingConfidence'] * 100))
    
    print('\n*** Likelihood ***')
    print('Joy: ' + annotation['joyLikelihood'])
    print('Sorrow: ' + annotation['sorrowLikelihood'])
    
    print('\n*** Features ***')
    if 'landmarks' in annotation:
        for features in annotation['landmarks']:
            print('\tType: ' + features['type'])
            coordinates = features['position']
            for key, value in coordinates.items():
                print('\t', key, value)
```

<details>
<summary>Response Analysis Details</summary>

#### Confidence Scores
- **Detection Confidence**: Certainty of face detection (0-1)
- **Landmarking Confidence**: Accuracy of facial feature positions (0-1)
- Values multiplied by 100 for percentage representation

#### Emotional Likelihood Values
- VERY_LIKELY
- LIKELY
- POSSIBLE
- UNLIKELY
- VERY_UNLIKELY

#### Facial Landmarks Detected
- Eyes (left and right)
- Eyebrows
- Nose tip and bridge
- Mouth (corners, center)
- Ears
- Chin
- Each landmark includes x, y, z coordinates relative to the image
</details>


# Drawing Facial Bounding Boxes

<details>
<summary>Background & Concepts</summary>

After detecting faces in images, visualizing the detection results is crucial for verification and analysis. Bounding boxes are rectangular regions that highlight detected faces, helping developers and users understand the API's detection accuracy and coverage.

**Boundary Types:**
- `boundingPoly`: Larger box including hair and accessories
- `fdBoundingPoly`: Tighter box focusing on facial skin area only
</details>

## Face Detection Code

### Understanding Boundaries
```python
annotations[0]['fdBoundingPoly']
```

### Drawing Helper Function
```python
def drawbox(image, left, top, right, bottom, text):
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, right, bottom], 
                  outline=(255,255,0,255))  # yellow line
    draw.rectangle([left, top, right, top + 12], 
                  fill=(255,255,0,255), 
                  outline=(255,255,0,255))
    draw.text((left, top), text, 
              fill=(0,0,0,255))  # black
```

### Drawing Implementation
```python
vertices = annotations[0]['fdBoundingPoly']['vertices']
    
image = Image.open(img_filename)

drawbox(image, 
        vertices[0]['x'], vertices[0]['y'], 
        vertices[2]['x'], vertices[2]['y'], 
        'Face 0')

plt.imshow(image)
plt.axis('off')
```

<details>
<summary>Implementation Details</summary>

**Color Values (RGBA):**
- Red (255), Green (255), Blue (0), Alpha (255) creates solid yellow
- Second rectangle creates solid background for text readability

**Vertex Information:**
- vertices[0]: Top-left corner
- vertices[1]: Top-right corner
- vertices[2]: Bottom-right corner
- vertices[3]: Bottom-left corner
- Coordinates in pixels relative to image dimensions
</details>

# Landmark Detection

## Setup and Request
```python
# Display image
img_filename1 = 'data/place.jpg'
plt.imshow(plt.imread(img_filename1))
plt.axis('off')

# Make API request
data = {
    'requests': [
        {
            'image': { 'content': get_base64(img_filename1).decode('utf-8') },
            'features': [{ 'type': 'LANDMARK_DETECTION' }]
        }
    ]
}

r = requests.post(url=googleurl, headers=req_headers, json=data)
```

## Response Processing
```python
# Helper function for vertex processing
def get_full_vertex(vertex):
    if not 'x' in vertex:
        vertex['x'] = 0
    if not 'y' in vertex:
        vertex['y'] = 0
    return vertex

# Draw landmark boundaries
image1 = Image.open(img_filename1)
for annotation in annotations_landmarks:
    vertices = annotation['boundingPoly']['vertices']
    
    drawbox(image1, 
        get_full_vertex(vertices[0])['x'], get_full_vertex(vertices[0])['y'], 
        get_full_vertex(vertices[2])['x'], get_full_vertex(vertices[2])['y'],
        annotation['description'])
    
plt.imshow(image1)
plt.axis('off')
```

<details>
<summary>Landmark Detection Details</summary>

**Response Information:**
- Description (landmark name)
- Confidence score
- Geographical coordinates
- Bounding polygon coordinates
- Associated metadata

**Error Handling:**
- `get_full_vertex()` provides default values (0) for missing coordinates
- Handles incomplete vertices at image boundaries
- Maintains visual representation with partial data

**Multiple Landmarks:**
- Code iterates through all detected landmarks
- Draws separate boxes with unique labels
- Maintains visual distinction between landmarks
</details>


# Handwriting Detection Guide

<details>
<summary>Background & Overview</summary>

Handwriting detection is a specialized form of optical character recognition (OCR) that focuses on interpreting and digitizing handwritten text. Google Cloud Vision API's DOCUMENT_TEXT_DETECTION feature provides advanced capabilities for processing both printed and handwritten text, making it valuable for:
- Document digitization
- Form processing
- Historical document preservation

**DOCUMENT_TEXT_DETECTION vs TEXT_DETECTION:**
- Better handling of document structure
- Improved accuracy for handwritten text
- Preservation of text layout information
- More detailed text block relationships
</details>

## Implementation

### Image Setup
```python
img_filename2 = 'data/note.jpg'
plt.imshow(plt.imread(img_filename2))
plt.axis('off');
```

### API Request
```python
data = {
    'requests': [
        {
            'image': { 'content': get_base64(img_filename2).decode('utf-8') },
            'features': [{ 'type': 'DOCUMENT_TEXT_DETECTION' }]
        }
    ]
}
r = requests.post(url=googleurl, headers=req_headers, json=data)
```

### Text Processing and Visualization
```python
# Load image and get annotations
image2 = Image.open(img_filename2)
annotations_text = result['responses'][0]['textAnnotations']

# Draw boundaries for each text element
for annotation in annotations_text:
    vertices = annotation['boundingPoly']['vertices']
    drawbox(image2, 
        get_full_vertex(vertices[0])['x'], get_full_vertex(vertices[0])['y'], 
        get_full_vertex(vertices[2])['x'], get_full_vertex(vertices[2])['y'],
        annotation['description'])
    
plt.imshow(image2)
plt.axis('off');
```

<details>
<summary>Response Structure Details</summary>

**API Response Contents:**
- Full text content
- Individual word boundaries
- Text block structures
- Confidence scores
- Spatial relationships between text elements

**Text Annotation Hierarchy:**
1. First element: Complete text from the image
2. Subsequent elements: Individual words/text blocks
3. Each annotation includes:
   - Description (text content)
   - Bounding polygon coordinates
   - Language identification (when applicable)

**Text Detection Levels:**
- Page level: Overall document structure
- Block level: Paragraphs and text sections
- Word level: Individual word boundaries
- Character level: Individual character positions
</details>

<details>
<summary>Best Practices & Tips</summary>

**For Optimal Results:**
- Ensure good image quality and resolution
- Maintain consistent lighting
- Avoid excessive rotation or skew
- Use clear handwriting with good contrast
- Consider image preprocessing for better results

**Common Applications:**
- Form digitization
- Note transcription
- Document archiving
- Historical document preservation
</details>
