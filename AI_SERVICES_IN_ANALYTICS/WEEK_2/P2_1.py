r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""
# |%%--%%| <XLRN1EXiQi|C6A7udMJdf>
r"""°°°
# Client Library
°°°"""
# |%%--%%| <C6A7udMJdf|XNjj4a1rC7>
r"""°°°
Previously, we have been able to perform vision detections using just an API key. In this exercise, we will use the Cloud Client Library (Python) with a service key to perform Vision and Video predictions.

Before we do that, we need to do some setup. Complete the following steps on your **PC/laptop**.
°°°"""
# |%%--%%| <XNjj4a1rC7|HdzdDeG6JP>
r"""°°°
### Set up PC

1. Download the [Google Cloud CLI installer](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe)
2. Double click to install the SDK
°°°"""
# |%%--%%| <HdzdDeG6JP|hE0fqc4c6C>
r"""°°°
3. Click Next. 
°°°"""
# |%%--%%| <hE0fqc4c6C|hBuT0fr79u>
r"""°°°
4. Click "I Agree"
5. Use default installation path
°°°"""
# |%%--%%| <hBuT0fr79u|B9Y0h52nm1>
r"""°°°
6. Click "Next" when done
°°°"""
# |%%--%%| <B9Y0h52nm1|i2UYJuNlRx>
r"""°°°
6. Check all the boxes and click "Finish"
7. The terminal window will auto launch gcloud init
8. Press Y
°°°"""
# |%%--%%| <i2UYJuNlRx|r6vJz7Dqv5>
r"""°°°
9. Your browser will launch. Sign in to your personal Google account. Agree to the terms of service.
10. Once done, press the Win button, search for <code>Google Cloud SDK Shell</code> and launch it
11. Type <code>gcloud auth list</code> to see your active account; it should be the account you have just logged in
12. Type <code>gcloud config list</code> to see some info on configuration
°°°"""
# |%%--%%| <r6vJz7Dqv5|YUfSWgsjWs>
r"""°°°
### Required files
°°°"""
# |%%--%%| <YUfSWgsjWs|QtRiH9uPRY>
r"""°°°
Before you proceed, download the following data from Brightspace into the current directory:
- <code>it3386-2024-s2.json</code>: service key
- images from week 1 practical
- <code>oh_2021_short.mp4</code>

You should have also completed the Anaconda environment set up so that you can import the vison and video intelligence libraries. See the instructions on Brightspace.
°°°"""
# |%%--%%| <QtRiH9uPRY|N8V2jA5KJg>
r"""°°°
Try performing a Vision Cloud Detection using the following codes.
°°°"""
#|%%--%%| <N8V2jA5KJg|V38OvgHZcM>

print("hello")

#|%%--%%| <V38OvgHZcM|91CfUZ1ssr>

%conda env list

# |%%--%%| <91CfUZ1ssr|zJgh3o4PVB>

from google.cloud import vision
import io

# |%%--%%| <zJgh3o4PVB|2m3cmLPncW>

# refer to https://cloud.google.com/vision/docs/detecting-faces

def detect_faces(path):
    """Detects faces in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
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

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message))
    
    return response

# |%%--%%| <2m3cmLPncW|M7MDA3tJtq>

# ensure the image is present in the current folder
response = detect_faces('nyp_cafe.jpg')

# |%%--%%| <M7MDA3tJtq|HTweAgjBhX>
r"""°°°
> Are you able to get a response from the detection? What is the error?
°°°"""
# |%%--%%| <HTweAgjBhX|FrSM6MuMUJ>



# |%%--%%| <FrSM6MuMUJ|tv8Ks3Ku9J>
r"""°°°
### Set up environment

The service key that you downloaded will give you the rights to complete the detections. Place them in the same directory.

We will set up the GOOGLE_APPLICATION_CREDENTIALS env to link to the service key.
°°°"""
# |%%--%%| <tv8Ks3Ku9J|Tf3Prgirnl>

# you should see the service key (json) in your current folder
!dir *.json

# |%%--%%| <Tf3Prgirnl|dZPr5SaCJS>

# env should be empty 
%env GOOGLE_APPLICATION_CREDENTIALS

# |%%--%%| <dZPr5SaCJS|vDWfF7Q3li>

# set the service key
%env GOOGLE_APPLICATION_CREDENTIALS=it3386-2024-s2.json

# |%%--%%| <vDWfF7Q3li|sJgLgmCMgi>

%env GOOGLE_APPLICATION_CREDENTIALS

# |%%--%%| <sJgLgmCMgi|Idv3VHmbHo>
r"""°°°
### Using Client Library for Vision
°°°"""
# |%%--%%| <Idv3VHmbHo|dpOlQ9pyOW>

response = detect_faces('../WEEK_1/data/nyp_cafe.jpg')

# |%%--%%| <dpOlQ9pyOW|qsfvLdOfba>

print(response)

# |%%--%%| <qsfvLdOfba|tc4g6COyJL>



# |%%--%%| <tc4g6COyJL|aXutlcBjtk>

def detect_landmarks(path):
    """Detects landmarks in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    print('Landmarks:')

    for landmark in landmarks:
        print(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print('Latitude {}'.format(lat_lng.latitude))
            print('Longitude {}'.format(lat_lng.longitude))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message))
        
    return response

# |%%--%%| <aXutlcBjtk|AwgzuBJNt9>

response = detect_landmarks('../WEEK_1/data/place.jpg')

# |%%--%%| <AwgzuBJNt9|kLkOUh525N>

print(response)

# |%%--%%| <kLkOUh525N|5GHmetkpdM>

def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message))
    
    return response

# |%%--%%| <5GHmetkpdM|I3nKipuZij>

response = detect_document('../WEEK_1/data/note.jpg')

# |%%--%%| <I3nKipuZij|5XDE3dr0aG>

print(response)

# |%%--%%| <5XDE3dr0aG|jXv3veLL9a>



# |%%--%%| <jXv3veLL9a|1SW3USoDky>
r"""°°°
Todo

> How about detecting labels?
°°°"""
# |%%--%%| <1SW3USoDky|NAp7BON4cJ>

# see https://cloud.google.com/vision/docs/labels

def detect_labels(path):

   

# |%%--%%| <NAp7BON4cJ|jkWKfbpPjF>

detect_labels('nyp_cafe.jpg')

# |%%--%%| <jkWKfbpPjF|SykwIOCbQz>
r"""°°°
*Sample output:*

<pre>
Labels:
Leisure
Customer
T-shirt
Eyewear
Event
Fun
Water bottle
Belt
Job
Room
</pre>
°°°"""
# |%%--%%| <SykwIOCbQz|hhJURIAxfz>



# |%%--%%| <hhJURIAxfz|Ji1xadC6xY>



# |%%--%%| <Ji1xadC6xY|WweEyR3FJf>


