r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""
# |%%--%%| <Ubv7HSlX5J|TB1FsWR66K>
r"""°°°
## Google Cloud Storage

In this practical, we are going to perform basic operations in Google buckets using the Google Client Library for Python and a service key. 
°°°"""
# |%%--%%| <TB1FsWR66K|tbQq0jr4eP>

# conda install -c conda-forge google-cloud-storage
!pip install google-cloud-storage

# |%%--%%| <tbQq0jr4eP|ixOcbKzLFb>

# Imports the Google Cloud client library
from google.cloud import storage

# |%%--%%| <ixOcbKzLFb|QV82rGDMnb>

# confirm service key is present in current folder
!dir *.json

# |%%--%%| <QV82rGDMnb|o8DopNtuB3>

%env GOOGLE_APPLICATION_CREDENTIALS

# |%%--%%| <o8DopNtuB3|1Z9vP4vfb0>

# fill in json filename
%env GOOGLE_APPLICATION_CREDENTIALS=WEEK_6/key_i220342h.json
%env GOOGLE_APPLICATION_CREDENTIALS

# |%%--%%| <1Z9vP4vfb0|uQ67eiOztc>

# settings
# refer to credentials sent in email
PROJECT_ID = "i220342h"
BUCKET_NAME = PROJECT_ID + "-3386-p7"

BUCKET_URI = f"gs://{BUCKET_NAME}"
REGION = "us-central1"
print(BUCKET_NAME)
print(BUCKET_URI)

# |%%--%%| <uQ67eiOztc|rnIFvxdFde>

# Instantiates a client
storage_client = storage.Client()

# |%%--%%| <rnIFvxdFde|CO1ErGUofV>
r"""°°°
### Create a new bucket
°°°"""
# |%%--%%| <CO1ErGUofV|P4PmoIHMjh>

#https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-client_libraries

my_bucket = storage_client.create_bucket(BUCKET_NAME)

# |%%--%%| <P4PmoIHMjh|J4z7OkN7nC>

# list buckets in project
buckets = storage_client.list_buckets()
for bucket in buckets:
    print(bucket.name)

# |%%--%%| <J4z7OkN7nC|9uVpNQwqAs>

# Lists all the blobs in the bucket
# empty for new bucket
blobs = storage_client.list_blobs(BUCKET_NAME)
for blob in blobs:
    print(blob.name)

# |%%--%%| <9uVpNQwqAs|DWXSonicrK>

# https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-client-libraries

def upload_directory_with_transfer_manager(bucket, source_directory, workers=8):
    """Upload every file in a directory, including all files in subdirectories.

    Each blob name is derived from the filename, not including the `directory`
    parameter itself. For complete control of the blob name for each file (and
    other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to upload. Files in the directory and its
    # subdirectories will be uploaded. An empty string means "the current
    # working directory".
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from pathlib import Path

    from google.cloud.storage import Client, transfer_manager

    # Generate a list of paths (in string form) relative to the `directory`.
    # This can be done in a single list comprehension, but is expanded into
    # multiple lines here for clarity.

    # First, recursively get all files in `directory` as Path objects.
    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")

    # Filter so the list only includes files, not directories themselves.
    file_paths = [path for path in paths if path.is_file()]

    # These paths are relative to the current working directory. Next, make them
    # relative to `directory`
    relative_paths = [path.relative_to(source_directory) for path in file_paths]

    # Finally, convert them all to strings.
    string_paths = [str(path) for path in relative_paths]

    print("Found {} files.".format(len(string_paths)))
    
    # filenames must in forward slashes to create the corresponding folder inside bucket
    string_paths[:] = [Path(path).as_posix() for path in string_paths]
    
    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket, string_paths, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(string_paths, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))
    

# |%%--%%| <DWXSonicrK|K0yaQHuSpv>
r"""°°°
### Upload folder and files to bucket

Steps
1. Create a folder (e.g. no_name; the folder name does not matter because it will not be reflected in the bucket)
2. Unzip masks.zip inside folder
°°°"""
# |%%--%%| <K0yaQHuSpv|KTyKFhoj1K>

!ls ./WEEK_6/no_name

# |%%--%%| <KTyKFhoj1K|BvaeuhyGaf>
r"""°°°
Sample Output

```
...

04/05/2024  06:33 PM    <DIR>          .
04/05/2024  06:33 PM    <DIR>          ..
04/05/2024  06:03 PM    <DIR>          masks
...
```
°°°"""
# |%%--%%| <BvaeuhyGaf|9xvBlgIoPS>

!ls ./WEEK_6/no_name/masks

# |%%--%%| <9xvBlgIoPS|kQg869VBLa>
r"""°°°
Sample Output
```
 ... \no_name\masks

04/05/2024  06:03 PM    <DIR>          .
04/05/2024  06:03 PM    <DIR>          ..
25/09/2020  09:52 AM            32,412 1.jpg
25/09/2020  09:52 AM            53,267 2.jpg
09/11/2023  03:52 PM               375 data.csv
...
```
°°°"""
# |%%--%%| <kQg869VBLa|BSMhV63NDa>

# e.g. s1-21264a-3386-p7
print(my_bucket.name)

# |%%--%%| <BSMhV63NDa|YngyupyB0M>

# upload all folder and files inside "no_name" folder to bucket
upload_directory_with_transfer_manager(my_bucket, './WEEK_6/no_name')

# |%%--%%| <YngyupyB0M|04EJEXiv3s>

# Lists all the blobs in the bucket
blobs = storage_client.list_blobs(BUCKET_NAME)
for blob in blobs:
    print(blob)

# |%%--%%| <04EJEXiv3s|eW8mzUnafT>

# checking content of a file in bucket

file_path = 'masks/data.csv'
blob = my_bucket.get_blob(file_path)
print(blob.download_as_string().decode())

# |%%--%%| <eW8mzUnafT|AMmNLGaykY>
r"""°°°
Sample output
```
gs://s1-21264a-aip-3386/masks/1.jpg,mask,0.22903225806451613,0.25806451612903225,,,0.3629032258064516,0.4792626728110599,,
gs://s1-21264a-aip-3386/masks/2.jpg,no-mask,0.7265795206971678,0.0691699604743083,,,0.7701525054466231,0.16996047430830039,,
gs://s1-21264a-aip-3386/masks/2.jpg,mask,0.8300653594771242,0.005928853754940711,,,0.8779956427015251,0.08300395256916997,,
```
°°°"""
# |%%--%%| <AMmNLGaykY|ItxK7xotVg>



# |%%--%%| <ItxK7xotVg|LTMiLCMVZn>
r"""°°°
### Delete Bucket

A bucket must be empty of objects before it can be deleted
°°°"""
# |%%--%%| <LTMiLCMVZn|CGe68lRj7F>

# delete objects from bucket
# https://cloud.google.com/storage/docs/deleting-objects#storage-delete-object-python

blobs = my_bucket.list_blobs()
for blob in blobs:
    blob.delete()
    print('{} deleted'.format(blob.name))

# |%%--%%| <CGe68lRj7F|Ni3Us6gmDu>

# delete a bucket https://cloud.google.com/storage/docs/deleting-buckets#delete-bucket-client-libraries

my_bucket.delete()

# |%%--%%| <Ni3Us6gmDu|Z3HUVigMgo>



# |%%--%%| <Z3HUVigMgo|VpPtGjXT2n>
r"""°°°
### References

- https://cloud.google.com/storage/docs/introduction
- https://cloud.google.com/storage/docs/samples/storage-upload-file
- https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
°°°"""
# |%%--%%| <VpPtGjXT2n|ROGpfo4lnN>



# |%%--%%| <ROGpfo4lnN|KD28aM5uHe>



# |%%--%%| <KD28aM5uHe|2vm3bgJNrA>



# |%%--%%| <2vm3bgJNrA|Ko8fO5Q4O4>



# |%%--%%| <Ko8fO5Q4O4|ycQZtNUTQ8>


