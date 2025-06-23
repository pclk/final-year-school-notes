from flask import Flask, request, jsonify
import torch
import os
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from torch import nn
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Define the model architecture
class SalaryPredictor(nn.Module):
    def __init__(self, n_hidden=768):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, n_hidden)  # 768 is DistilBERT's hidden size
        self.relu = nn.ReLU()
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output = self.drop(pooled_output)
        output = self.fc(output)
        output = self.relu(output)
        output = self.out(output)
        return output


# Global variables for model components
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TOKENIZER = None
SCALER = None


def load_model():
    """Initialize all model components"""
    global MODEL, TOKENIZER, SCALER

    try:
        # Define model paths
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "best_model.pt")
        scaler_path = os.path.join(model_dir, "salary_scaler.joblib")

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

        # Initialize model
        logger.info("Loading BERT model...")
        MODEL = SalaryPredictor()
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()

        # Load tokenizer and scaler
        logger.info("Loading tokenizer...")
        TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        logger.info("Loading scaler...")
        SCALER = joblib.load(scaler_path)

        logger.info("Model, tokenizer, and scaler loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return False


# Load model components at app initialization
logger.info("Loading model components during app initialization...")
load_model()


def predict_salary(input_text):
    """Make salary prediction using the loaded model"""
    with torch.no_grad():
        # Tokenize input
        encoding = TOKENIZER.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        # Get prediction
        output = MODEL(input_ids=input_ids, attention_mask=attention_mask)

        # Move output to CPU and convert to numpy
        output = output.cpu().numpy()

        # Convert prediction back to salary scale
        prediction = SCALER.inverse_transform(output)[0][0]

        return prediction


@app.route("/predict", methods=["POST"])
def predict():
    try:
        logger.info("Prediction endpoint called")
        # Check if model is loaded
        if MODEL is None:
            logger.warning("Model not loaded, attempting to load...")
            if not load_model():
                logger.error("Failed to load model components")
                return jsonify({"error": "Failed to load model components"}), 500

        # Get JSON data from request
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")

        # Format input text
        input_text = f"Job Title: {data.get('job_title', '')} "
        input_text += f"Query: {data.get('query', '')} "
        input_text += f"Description: {data.get('job_description', '')} "
        input_text += f"Location: {data.get('location', '')} "
        input_text += f"Country: {data.get('country', '')} "
        input_text += f"Contract: {data.get('contract_type', '')} "
        input_text += f"Education: {data.get('education_level', '')} "
        input_text += f"Seniority: {data.get('seniority', '')} "
        input_text += f"Experience: {data.get('min_years_experience', '0')} years"

        # Get prediction
        predicted_salary = predict_salary(input_text)

        return jsonify({"predicted_salary": float(predicted_salary)})

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/health", methods=["GET"])
def health_check():
    logger.info("Health check endpoint called")
    status = "healthy" if all([MODEL, TOKENIZER, SCALER]) else "unhealthy"
    logger.info(f"Health status: {status}")
    return jsonify(
        {
            "status": status,
            "model_loaded": MODEL is not None,
            "tokenizer_loaded": TOKENIZER is not None,
            "scaler_loaded": SCALER is not None,
        }
    )


if __name__ == "__main__":
    # Load model at startup
    print("Loading model components...")
    load_model()

    # Run app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
