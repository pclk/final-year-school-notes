1. **Upload Model to Cloud Storage**
   - First, ensure your SavedModel is uploaded to a Google Cloud Storage bucket
   - The model should be in the SavedModel format (the directory containing saved_model.pb)

2. **Import Model to Vertex AI**
   - Go to Vertex AI section in Google Cloud Console
   - Navigate to "Model Registry" in the left sidebar
   - Click "Import" button
   - Fill in the following:
     ```
     Name: salary-prediction-model
     Model settings: Import model artifacts into Vertex AI
     Model artifact format: TensorFlow SavedModel
     Model artifact location: Browse to your GCS bucket location
     Model framework: TensorFlow 2.x
     ```

3. **Create Endpoint**
   - Go to "Endpoints" in the left sidebar
   - Click "Create Endpoint"
   - Fill in:
     ```
     Name: salary-prediction-endpoint
     Region: Select your desired region (e.g., us-central1)
     ```

4. **Deploy Model to Endpoint**
   - After creating the endpoint, click "Deploy Model"
   - Select your imported model
   - Configure the following settings:

   ```
   Traffic settings:
   - Traffic percentage: 100%

   Machine configuration:
   - Machine type: n1-standard-2 (2 vCPU, 7.5 GB memory)
   - GPU: None (unless you need it)
   - Service account: Default compute service account

   Scaling settings:
   - Minimum number of nodes: 1
   - Maximum number of nodes: 1
   - CPU utilization target: 60%

   Container settings:
   - Container image: us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest
   - Port: 8080
   - Health check: /v1/models/${MODEL_NAME}
   ```

5. **Important Configuration Notes**:
   - The container image version (tf2-cpu.2-12) should match your TensorFlow version
   - For your model, use the TensorFlow 2.x CPU-based container
   - Make sure your service account has necessary permissions:
     - roles/aiplatform.user
     - roles/storage.objectViewer

6. **After Deployment**
   - The endpoint will show "Deploying" status
   - Wait until it shows "Deployed" (may take several minutes)
   - You can then use the endpoint ID for predictions

7. **Testing the Endpoint**
   - In the endpoint details, there's a "Test" tab
   - You can test with this JSON format:
   ```json
   {
     "instances": [{
       "job_description_input": "Software engineer position...",
       "job_title_input": "Software Engineer",
       "query_input": "software engineer",
       "soft_skills_input": ["communication", "teamwork"],
       "hard_skills_input": ["python", "javascript"],
       "location_flexibility_input": "remote",
       "contract_type_input": "full_time",
       "education_level_input": "bachelor",
       "seniority_input": "mid",
       "min_years_experience_input": 3,
       "field_of_study_input": ["computer science"]
     }]
   }
