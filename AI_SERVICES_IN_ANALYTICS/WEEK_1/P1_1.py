r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""

# |%%--%%| <ZB7NmFI0BI|nGg1xfX8eT>
r"""°°°
## Google Cloud Vision API

In this practical, we are going to learn more about the Google Cloud Vision API. You have some image files in the data directory that you can work with but feel free to try out your own images anytime.

°°°"""
# |%%--%%| <nGg1xfX8eT|SI9ZcbQQr2>
r"""°°°
Let's start with exploring the [Cloud Vision Demo](https://cloud.google.com/vision).

Go to the link. Under the demo section, upload an image. If you upload <code>nyp_cafe.jpg</code>, you will see the following.
°°°"""
# |%%--%%| <SI9ZcbQQr2|4tM8PWRwEe>
r"""°°°
![detection_face_s.png](attachment:detection_face_s.png)
°°°"""
# |%%--%%| <4tM8PWRwEe|iOhNF55avN>
r"""°°°
### Todo

> Upload any image and observe the results of the various detections.
°°°"""
# |%%--%%| <iOhNF55avN|nCUnx4zzX5>
r"""°°°
### Face Detection

Cloud Vision API supports many types of detection. Let's start with face detection.

We will now connect to the Cloud Vision API to make a request and get a response.
°°°"""
# |%%--%%| <nCUnx4zzX5|sTTCkyzxbQ>

import base64
import json

import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageDraw

# |%%--%%| <sTTCkyzxbQ|JT4BwZ7QCn>
r"""°°°
These parameters are required to complete the request.
°°°"""
# |%%--%%| <JT4BwZ7QCn|6fG4w2rtt6>

googleAPIKey = "AIzaSyBhNkvLI0d7ZcHxP95h7slBOs-5OT8nLa0"
googleurl = "https://vision.googleapis.com/v1/images:annotate?key=" + googleAPIKey
req_headers = {"Content-Type": "application/json"}


# |%%--%%| <6fG4w2rtt6|LQzXzu3Gln>


# helper function
def get_base64(image_filename):
    with open(image_filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


# |%%--%%| <LQzXzu3Gln|FbASNi5dIQ>
r"""°°°
This is the image we will work with.
°°°"""
# |%%--%%| <FbASNi5dIQ|78OFpMrUtS>

img_filename = "./data/nyp_cafe.jpg"

plt.imshow(plt.imread(img_filename))
plt.axis("off")
# |%%--%%| <78OFpMrUtS|O2hBJ68LP2>
r"""°°°
Make an API request. Ensure the request parameters are filled in correctly as required under [Cloud Vision Face Detection](https://cloud.google.com/vision/docs/detecting-faces#vision_face_detection-drest).
°°°"""
# |%%--%%| <O2hBJ68LP2|TyVR2uMdla>


data = {
    "requests": [
        {
            "image": {"content": get_base64(img_filename).decode("utf-8")},
            "features": [
                {"type": "FACE_DETECTION"},
                # { 'type': 'LANDMARK_DETECTION' }
            ],
        }
    ]
}

# Send the image data to Google for label detection
r = requests.post(url=googleurl, headers=req_headers, json=data)

# Check and display the results
if r.status_code == 200:
    result = r.json()

    print(result)

    # loop through the response to get the parameters needed


else:
    print("Error with status")
    print(r.content)


# |%%--%%| <TyVR2uMdla|FB9XHdw2iu>
r"""°°°
**Brief Analysis**

Using the result from the API, we can print out the content to understand it better and dive into the specific parameters. Do check if a certain key exist before accessing the dictionary to prevent your app from crashing.
°°°"""
# |%%--%%| <FB9XHdw2iu|widUhUcbUK>

# Pretty print JSON response
print(json.dumps(result, indent=4))

# |%%--%%| <widUhUcbUK|7GEUB0cfc3>
r"""°°°
Let's analyse the annotations in the response.
°°°"""
# |%%--%%| <7GEUB0cfc3|wGuXpVP95x>

# number of faces detected
annotations = result["responses"][0]["faceAnnotations"]
len(annotations)

# |%%--%%| <wGuXpVP95x|iQ3MK4d2De>

for annotation in annotations:
    print(json.dumps(annotation, indent=4))

# |%%--%%| <iQ3MK4d2De|2xIK8Q5gM6>
r"""°°°
Print out values of description and score properties
°°°"""
# |%%--%%| <2xIK8Q5gM6|fQWw7yGZQe>

for annotation in annotations:
    print("\n\nEach face")

    print("\n*** Confidence ***")
    print("Detection Confidence: %.2f" % (annotation["detectionConfidence"] * 100))
    print("Landmarking Confidence: %.2f" % (annotation["landmarkingConfidence"] * 100))

    print("\n*** Likelihood ***")
    print("Joy: " + annotation["joyLikelihood"])
    print("Sorrow: " + annotation["sorrowLikelihood"])

    print("\n*** Features ***")

    # check if key is present before processing
    if "landmarks" in annotation:
        for features in annotation["landmarks"]:
            print(features)

            # process each individual feature; uncomment to see details
            print("\tType: " + features["type"])
            coordinates = features["position"]
            for key, value in coordinates.items():
                print("\t", key, value)


# |%%--%%| <fQWw7yGZQe|VgzbYYx8N5>
r"""°°°
Let's plot a bounding polygon of the first face in the image. Note the definitions:
- `boundingPoly`: The bounding polygon around the face.
- `fdBoundingPoly`: This bounding polygon is tighter than the `boundingPoly`, and encloses only the skin part of the face.
°°°"""
# |%%--%%| <VgzbYYx8N5|BTzbPkJZ1l>
r"""°°°
These are the vertices of the first face (hence index 0).
°°°"""
# |%%--%%| <BTzbPkJZ1l|ME9VAhjW3r>

annotations[0]["fdBoundingPoly"]

# |%%--%%| <ME9VAhjW3r|U3mq1GgCK1>
r"""°°°
Let's draw the bounding box of the face on the image.
°°°"""
# |%%--%%| <U3mq1GgCK1|kaSRZxWGZx>


# helper function
def drawbox(image, left, top, right, bottom, text):
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [left, top, right, bottom], outline=(255, 255, 0, 255)
    )  # yellow line
    draw.rectangle(
        [left, top, right, top + 12],
        fill=(255, 255, 0, 255),
        outline=(255, 255, 0, 255),
    )
    draw.text((left, top), text, fill=(0, 0, 0, 255))  # black


# |%%--%%| <kaSRZxWGZx|BQTQj7xKes>

vertices = annotations[0]["fdBoundingPoly"]["vertices"]

image = Image.open(img_filename)

drawbox(
    image,
    vertices[0]["x"],
    vertices[0]["y"],
    vertices[2]["x"],
    vertices[2]["y"],
    "Face 0",
)

plt.imshow(image)
plt.axis("off")
# |%%--%%| <BQTQj7xKes|Bgxqz1AMmS>
r"""°°°
Details on vertices is shown below. You just need the top left (index 0) and bottom right (index 2) to plot a bounding box.
°°°"""
# |%%--%%| <Bgxqz1AMmS|MFhOYvPj1x>
r"""°°°
![vertices.JPG](attachment:vertices.JPG)
°°°"""
# |%%--%%| <MFhOYvPj1x|2GymRcZZMv>


# |%%--%%| <2GymRcZZMv|gGspXyOora>
r"""°°°
### Landmark Detection

Read up on [Landmark detection](https://cloud.google.com/vision/docs/detecting-landmarks).

Using the same technique above, perform landmark detection with place.jpg (<span class="attribution">"<a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/photos/57785759@N06/5552134623">Marina Bay Sands</a>" by <a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/photos/57785759@N06">alantankenghoe</a> is licensed under <a target="_blank" rel="noopener noreferrer" href="https://creativecommons.org/licenses/by/2.0/?ref=openverse">CC BY 2.0</a>.</span>)
°°°"""
# |%%--%%| <gGspXyOora|949JnZdtDj>
r"""°°°
Plot the bounding box with the description as shown below.
°°°"""
# |%%--%%| <949JnZdtDj|WEpWObeXvq>
r"""°°°
![image.png](attachment:image.png)
°°°"""
# |%%--%%| <WEpWObeXvq|0uwGie3H3i>
r"""°°°
> You might encounter the following error if you call the drawbox function:  
> `KeyError: 'x'`  
> 
> Why? Analyse the vertices. How do you resolve the problem?

°°°"""
# |%%--%%| <0uwGie3H3i|gV4kd6TcVN>
# Showing image
img_filename = "data/place.jpg"

plt.imshow(plt.imread(img_filename))
plt.axis("off")

# |%%--%%| <gV4kd6TcVN|gKZ33fDVD0>

# Making request
data = {
    "requests": [
        {
            "image": {"content": get_base64(img_filename).decode("utf-8")},
            "features": [
                # {"type": "FACE_DETECTION"},
                {"type": "LANDMARK_DETECTION"}
            ],
        }
    ]
}

r = requests.post(url=googleurl, headers=req_headers, json=data)

# Check and display the results
if r.status_code == 200:
    result = r.json()
    print(json.dumps(result, indent=4))

else:
    print("Error with status")

# |%%--%%| <gKZ33fDVD0|ulsh2f5mYo>

# drawing the annotations
vertices = result["responses"][0]["landmarkAnnotations"][0]["boundingPoly"]["vertices"]

image = Image.open(img_filename)

# top left x and y is 0
drawbox(
    image,
    vertices[0].get("x") or 0,
    vertices[0].get("y") or 0,
    vertices[2].get("x") or 0,
    vertices[2].get("y") or 0,
    "Location 0",
)

plt.imshow(image)
plt.axis("off")

# |%%--%%| <ulsh2f5mYo|G4ah7NeVET>
r"""°°°
### Handwriting Detection

Read up on [Handwriting detection](https://cloud.google.com/vision/docs/handwriting). 

Plot the bounding boxes with the descriptions as shown below. Perform landmark detection using note.jpg.
°°°"""
# |%%--%%| <G4ah7NeVET|iKLhpY6I13>
r"""°°°
![image.png](attachment:image.png)
°°°"""
# |%%--%%| <iKLhpY6I13|ilkHsNpbpT>

img_filename = "data/note.jpg"
data = {
    "requests": [
        {
            "image": {"content": get_base64(img_filename).decode("utf-8")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }
    ]
}

r = requests.post(url=googleurl, headers=req_headers, json=data)

# Check and display the results
if r.status_code == 200:
    result = r.json()
    print(json.dumps(result, indent=4))

else:
    print("Error with status")

# drawing the annotations
blocks = result["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"]

image = Image.open(img_filename)

for block in blocks:
    vertices = block["boundingBox"]["vertices"]
    drawbox(
        image,
        vertices[0].get("x") or 0,
        vertices[0].get("y") or 0,
        vertices[2].get("x") or 0,
        vertices[2].get("y") or 0,
        "Handwriting 0",
    )

plt.imshow(image)
plt.axis("off")

# |%%--%%| <ilkHsNpbpT|I0a6U50njY>
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
# |%%--%%| <I0a6U50njY|kOE7ioRkk9>
r"""°°°
> What applications can you build using these image detection capabilities?
°°°"""
# |%%--%%| <kOE7ioRkk9|3XxCoCNgdq>


# |%%--%%| <3XxCoCNgdq|rb3BSH5Kqv>
