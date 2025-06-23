Step 1: Refer to my files and replicate it.
Step 2: Move your models to the same folder as your flask app.
Step 3: Run the following commands inside the same directory as your flask app

```sh
gcloud init
```

```sh

gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

```sh
gcloud builds submit --tag gcr.io/rare-ethos-403914/salary-prediction
```
```sh
gcloud run deploy salary-prediction \
  --image gcr.io/rare-ethos-403914/salary-prediction \
  --platform managed \
  --region asia-southeast1 \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated \
  --min-instances 1
```

Testing locally:
run the following commands

```sh
docker build -t "app" .
```
```sh
docker run -p 8080:8080 -e PORT=8080 app
```

include median salary in genai system prompt

show pretrained model
