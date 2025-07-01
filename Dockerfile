# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the container
COPY src/ ./src/
# Copy data.csv if your API's preprocessing logic needs access to it (e.g., for RFM calculations)
# For this specific API setup, it assumes preprocessed input, so data.csv might not be strictly needed
# unless you change the API to do full preprocessing internally.
# COPY data.csv .

# Set environment variables for MLflow (if using a remote server)
# ENV MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
# ENV MLFLOW_S3_ENDPOINT_URL="http://your-minio-server:9000" # If using S3/MinIO for artifacts

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]