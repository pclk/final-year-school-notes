r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""
# |%%--%%| <3hsveRWqaL|jEBuNzVDP5>
r"""°°°
## Google Cloud Vision API

In this practical, we are going to learn more about the Google Cloud Vision API. You have some image files in the data directory that you can work with but feel free to try out your own images anytime.

°°°"""
# |%%--%%| <jEBuNzVDP5|tk097OARtg>
r"""°°°
Let's start with exploring the [Cloud Vision Demo](https://cloud.google.com/vision).

Go to the link. Under the demo section, upload an image. If you upload <code>nyp_cafe.jpg</code>, you will see the following.
°°°"""
# |%%--%%| <tk097OARtg|33UJ5qICvJ>
r"""°°°
![detection_face_s.png](attachment:detection_face_s.png)
°°°"""
# |%%--%%| <33UJ5qICvJ|i7nhb3NcOn>
r"""°°°
### Todo

> Upload any image and observe the results of the various detections.
°°°"""
# |%%--%%| <i7nhb3NcOn|R7erjkQkAn>
r"""°°°
### Face Detection

Cloud Vision API supports many types of detection. Let's start with face detection.

We will now connect to the Cloud Vision API to make a request and get a response.
°°°"""
# |%%--%%| <R7erjkQkAn|4A6RZOvLpx>

import requests 
import base64
import json
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
%matplotlib inline


# |%%--%%| <4A6RZOvLpx|3WLXE8kk1F>
r"""°°°
These parameters are required to complete the request.
°°°"""
# |%%--%%| <3WLXE8kk1F|Mt33wHoYti>

googleAPIKey = "AIzaSyBhNkvLI0d7ZcHxP95h7slBOs-5OT8nLa0"
googleurl = "https://vision.googleapis.com/v1/images:annotate?key=" + googleAPIKey
req_headers = {'Content-Type': 'application/json'}


# |%%--%%| <Mt33wHoYti|J3HLKrSfk1>

# helper function
def get_base64(image_filename):
    with open(image_filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string

# |%%--%%| <J3HLKrSfk1|tSJleQkTat>
r"""°°°
This is the image we will work with.
°°°"""
# |%%--%%| <tSJleQkTat|P97uG9WBJk>

img_filename = 'data/nyp_cafe.jpg'

plt.imshow(plt.imread(img_filename))
plt.axis('off');


# |%%--%%| <P97uG9WBJk|xmAvQ2jOFG>
r"""°°°
Make an API request. Ensure the request parameters are filled in correctly as required under [Cloud Vision Face Detection](https://cloud.google.com/vision/docs/detecting-faces#vision_face_detection-drest).
°°°"""
# |%%--%%| <xmAvQ2jOFG|VmMhSs16Jh>


data = {
    'requests': 
    [
        {
            'image': { 'content': get_base64(img_filename).decode('utf-8') },
            'features': [{ 'type': 'FACE_DETECTION' },
                         #{ 'type': 'LANDMARK_DETECTION' }
                        ]
        }
    ]
}

# Send the image data to Google for label detection
r = requests.post(url = googleurl, headers = req_headers, json = data) 

# Check and display the results
if r.status_code == 200:
    result = r.json()

    print (result)

    # loop through the response to get the parameters needed
    
    
else:
    print('Error with status')
    print(r.content)
    


# |%%--%%| <VmMhSs16Jh|VpsRJsgZzW>
r"""°°°
**Brief Analysis**

Using the result from the API, we can print out the content to understand it better and dive into the specific parameters. Do check if a certain key exist before accessing the dictionary to prevent your app from crashing.
°°°"""
# |%%--%%| <VpsRJsgZzW|vmdG7BoXLz>

# Pretty print JSON response
print(json.dumps(result, indent=4))

# |%%--%%| <vmdG7BoXLz|6OqPxSjOGU>
r"""°°°
Let's analyse the annotations in the response.
°°°"""
# |%%--%%| <6OqPxSjOGU|vihj6W2OWE>

# number of faces detected
annotations = result['responses'][0]['faceAnnotations']
len(annotations)

# |%%--%%| <vihj6W2OWE|K7wDYHrsCn>

for annotation in annotations:
    print(json.dumps(annotation, indent=4))

# |%%--%%| <K7wDYHrsCn|p9tikAqMIy>
r"""°°°
Print out values of description and score properties
°°°"""
# |%%--%%| <p9tikAqMIy|NHIb6XZKwy>

for annotation in annotations:
    print('\n\nEach face')
    
    print('\n*** Confidence ***')
    print('Detection Confidence: %.2f' % (annotation['detectionConfidence'] * 100))
    print('Landmarking Confidence: %.2f' % (annotation['landmarkingConfidence'] * 100))
    
    print('\n*** Likelihood ***')
    print('Joy: ' + annotation['joyLikelihood'])
    print('Sorrow: ' + annotation['sorrowLikelihood'])
    
    print('\n*** Features ***')
    
    # check if key is present before processing
    if 'landmarks' in annotation:
        for features in annotation['landmarks']:
            print(features)
            
            # process each individual feature; uncomment to see details
            print('\tType: ' + features['type'])
            coordinates = features['position']
            for key, value in coordinates.items():
                print('\t', key, value)
            
             
    

# |%%--%%| <NHIb6XZKwy|CT0MDnLYJ5>
r"""°°°
Let's plot a bounding polygon of the first face in the image. Note the definitions:
- `boundingPoly`: The bounding polygon around the face.
- `fdBoundingPoly`: This bounding polygon is tighter than the `boundingPoly`, and encloses only the skin part of the face.
°°°"""
# |%%--%%| <CT0MDnLYJ5|nVEDbxlJlQ>
r"""°°°
These are the vertices of the first face (hence index 0).
°°°"""
# |%%--%%| <nVEDbxlJlQ|Hk5rtm0uhw>

annotations[0]['fdBoundingPoly']

# |%%--%%| <Hk5rtm0uhw|1Z8BQNcqu5>
r"""°°°
Let's draw the bounding box of the face on the image.
°°°"""
# |%%--%%| <1Z8BQNcqu5|yrFgv1Qvz8>

# helper function
def drawbox(image, left, top, right, bottom, text):
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, right, bottom], outline=(255,255,0,255)) # yellow line
    draw.rectangle([left, top, right, top + 12], fill=(255,255,0,255), outline=(255,255,0,255))
    draw.text((left, top), text, fill=(0,0,0,255)) # black
    

# |%%--%%| <yrFgv1Qvz8|DI2qmKU1MM>

vertices = annotations[0]['fdBoundingPoly']['vertices']
    
image = Image.open(img_filename)

drawbox(image, 
        vertices[0]['x'], vertices[0]['y'], 
        vertices[2]['x'], vertices[2]['y'], 'Face 0')

plt.imshow(image)
plt.axis('off');


# |%%--%%| <DI2qmKU1MM|rL4YGTxn9v>
r"""°°°
Details on vertices is shown below. You just need the top left (index 0) and bottom right (index 2) to plot a bounding box.
°°°"""
# |%%--%%| <rL4YGTxn9v|KaKcrRktAC>
r"""°°°
![vertices.JPG](attachment:vertices.JPG)
°°°"""
# |%%--%%| <KaKcrRktAC|NreNBwdYTG>



# |%%--%%| <NreNBwdYTG|9YGFdynNIu>
r"""°°°
### Landmark Detection

Read up on [Landmark detection](https://cloud.google.com/vision/docs/detecting-landmarks).

Using the same technique above, perform landmark detection with place.jpg (<span class="attribution">"<a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/photos/57785759@N06/5552134623">Marina Bay Sands</a>" by <a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/photos/57785759@N06">alantankenghoe</a> is licensed under <a target="_blank" rel="noopener noreferrer" href="https://creativecommons.org/licenses/by/2.0/?ref=openverse">CC BY 2.0</a>.</span>)
°°°"""
# |%%--%%| <9YGFdynNIu|imph4t6aUD>
r"""°°°
Plot the bounding box with the description as shown below.
°°°"""
# |%%--%%| <imph4t6aUD|7CwWmMWd3c>
r"""°°°
![image.png](attachment:image.png)
°°°"""
# |%%--%%| <7CwWmMWd3c|4tAFIqErJk>
r"""°°°
> You might encounter the following error if you call the drawbox function:  
> `KeyError: 'x'`  
> 
> Why? Analyse the vertices. How do you resolve the problem?

°°°"""
# |%%--%%| <4tAFIqErJk|Z0yxK0vi0q>

img_filename1 = 'data/place.jpg'
plt.imshow(plt.imread(img_filename1))
plt.axis('off');

# |%%--%%| <Z0yxK0vi0q|nXygcg2B91>


data = {
    'requests': 
    [
        {
            'image': { 'content': get_base64(img_filename1).decode('utf-8') },
            'features': [{ 'type': 'LANDMARK_DETECTION' }]
        }
    ]
}

# Send the image data to Google for label detection
r = requests.post(url = googleurl, headers = req_headers, json = data) 

# Check and display the results
if r.status_code == 200:
    result = r.json()

    print (result)

    # loop through the response to get the parameters needed
    
    
else:
    print('Error with status')
    print(r.content)
    


# |%%--%%| <nXygcg2B91|dBPBv0ZDPG>

# number of descriptions
len(result['responses'][0]['landmarkAnnotations'])

# |%%--%%| <dBPBv0ZDPG|EN2IwMmrZe>

result['responses'][0]['landmarkAnnotations'][0]['description']

# |%%--%%| <EN2IwMmrZe|SYZdEFnoKV>

image1 = Image.open(img_filename1)
annotations_landmarks = result['responses'][0]['landmarkAnnotations']
annotations_landmarks

# |%%--%%| <SYZdEFnoKV|vzr9Yi7aeU>

# example input {'x':100}
# if x or y key not found, set value to 0
# return proper vertex {'x':100, 'y':0}
def get_full_vertex(vertex):
    if not 'x' in vertex:
        vertex['x'] = 0
    if not 'y' in vertex:
        vertex['y'] = 0
    return vertex

# |%%--%%| <vzr9Yi7aeU|tV3grlZoM8>

for annotation in annotations_landmarks:
    vertices = annotation['boundingPoly']['vertices']
    
    # possible that x or y is not defined
    drawbox(image1, 
        get_full_vertex(vertices[0])['x'], get_full_vertex(vertices[0])['y'], 
        get_full_vertex(vertices[2])['x'], get_full_vertex(vertices[2])['y'],
        annotation['description'])
    
plt.imshow(image1)
plt.axis('off');


# |%%--%%| <tV3grlZoM8|qNHBYGgOnR>



# |%%--%%| <qNHBYGgOnR|DlvpuXYJIM>



# |%%--%%| <DlvpuXYJIM|OZwm4pMWgO>
r"""°°°
### Handwriting Detection

Read up on [Handwriting detection](https://cloud.google.com/vision/docs/handwriting). 

Plot the bounding boxes with the descriptions as shown below. Perform landmark detection using note.jpg.
°°°"""
# |%%--%%| <OZwm4pMWgO|Cp2vGxwGaO>
r"""°°°
![image.png](attachment:image.png)
°°°"""
# |%%--%%| <Cp2vGxwGaO|gOuicwmJ4b>



# |%%--%%| <gOuicwmJ4b|vUxfxG1JFp>

img_filename2 = 'data/note.jpg'
plt.imshow(plt.imread(img_filename2))
plt.axis('off');

# |%%--%%| <vUxfxG1JFp|29rVscaY6L>


data = {
    'requests': 
    [
        {
            'image': { 'content': get_base64(img_filename2).decode('utf-8') },
            'features': [{ 'type': 'DOCUMENT_TEXT_DETECTION' }]
        }
    ]
}

# Send the image data to Google for label detection
r = requests.post(url = googleurl, headers = req_headers, json = data) 

# Check and display the results
if r.status_code == 200:
    result = r.json()

    print (result)

    # loop through the response to get the parameters needed
    
    
else:
    print('Error with status')
    print(r.content)
    


# |%%--%%| <29rVscaY6L|O1tTeZtKvj>

image2 = Image.open(img_filename2)
annotations_text = result['responses'][0]['textAnnotations']
annotations_text

# |%%--%%| <O1tTeZtKvj|0Iqsx8L8kg>

for annotation in annotations_text:
    vertices = annotation['boundingPoly']['vertices']
    drawbox(image2, 
        get_full_vertex(vertices[0])['x'], get_full_vertex(vertices[0])['y'], 
        get_full_vertex(vertices[2])['x'], get_full_vertex(vertices[2])['y'],
        annotation['description'])
    
plt.imshow(image2)
plt.axis('off');

# |%%--%%| <0Iqsx8L8kg|OlQlc2h9Hy>



# |%%--%%| <OlQlc2h9Hy|cXSzqvYtkc>



# |%%--%%| <cXSzqvYtkc|rVhQox9kBl>
r"""°°°
### Miscellaneous

Try out these detections using Cloud Vision API
- [Text detection in images](https://cloud.google.com/vision/docs/ocr)
- [Image properties detection](https://cloud.google.com/vision/docs/detecting-properties)
- [Label detection](https://cloud.google.com/vision/docs/labels)
- [Logo detection](https://cloud.google.com/vision/docs/detecting-logos)

For each API, using an appropriate image, show that you can 
- send the correct request
- receive the intended response
- draw bounding boxes if the annotations in the response contain coordinates

°°°"""
# |%%--%%| <rVhQox9kBl|0QTymhk5p6>
r"""°°°
> What applications can you build using these image detection capabilities?
°°°"""
# |%%--%%| <0QTymhk5p6|IJI7EkFfcq>



# |%%--%%| <IJI7EkFfcq|vYhseqeIMu>


