# Use the official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 8080 available
EXPOSE 8080

# Run the application with Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
