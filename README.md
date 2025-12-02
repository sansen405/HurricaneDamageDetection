# COE Project 2 - Inference Server Deployment

## Overview

This project provides a Docker containerized inference server for image classification. The server uses a trained deep learning model to classify images as either "damage" or "no_damage". The server exposes two REST API endpoints: `/summary` for model metadata and `/inference` for image classification.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Image](#docker-image)
3. [Docker Compose Setup](#docker-compose-setup)
4. [API Endpoints](#api-endpoints)
5. [Example Requests](#example-requests)
6. [Running the Server](#running-the-server)
7. [Testing with the Grader](#testing-with-the-grader)
8. [Troubleshooting](#troubleshooting)

---

## Docker Image

### Image Information

- **Image Name:** `ssenthil99/inference-server:latest`
- **Architecture:** x86_64
- **Base Image:** Python 3.9-slim
- **Port:** 5000
- **Docker Hub:** https://hub.docker.com/r/ssenthil99/inference-server

### Pulling the Image

To pull the pre-built image from Docker Hub:

```bash
docker pull ssenthil99/inference-server:latest
```

### Building the Image Locally

If you need to build the image from source:

```bash
# Navigate to the project directory
cd COE_Project2

# Build the image
docker build -t ssenthil99/inference-server:latest .

# Or build without cache (for clean rebuild)
docker build --no-cache -t ssenthil99/inference-server:latest .
```

### Image Contents

The Docker image includes:
- Python 3.9 runtime
- Flask web framework
- TensorFlow/Keras for model inference
- PIL/Pillow for image processing
- NumPy for numerical operations
- The trained model (`best_model.keras`)
- Model metadata and preprocessing configuration
- The inference server application (`inference_server.py`)

---

## Docker Compose Setup

### Docker Compose File

The project includes a `docker-compose.yml` file for easy deployment:

```yaml
services:
  inference-server:
    image: ssenthil99/inference-server:latest
    ports:
      - "5000:5000"
    restart: unless-stopped
    environment:
      - FLASK_ENV=production
```

### Using Docker Compose

#### Start the Server

Start the inference server in detached mode (background):

```bash
docker-compose up -d
```

#### Stop the Server

Stop and remove the container:

```bash
docker-compose down
```

#### View Logs

View real-time logs from the server:

```bash
docker-compose logs -f
```

View logs without following:

```bash
docker-compose logs
```

#### Check Server Status

Check if the container is running:

```bash
docker-compose ps
```

#### Restart the Server

Restart the running container:

```bash
docker-compose restart
```

#### Rebuild and Restart

If you've updated the image or need to pull the latest version:

```bash
docker-compose down
docker-compose pull
docker-compose up -d
```

---

## API Endpoints

### 1. GET /summary

Returns metadata about the loaded model.

**Request:**
```http
GET /summary HTTP/1.1
Host: localhost:5000
```

**Response:**
```json
{
  "model_name": "Alternate LeNet-5",
  "test_auc": 0.9304,
  "input_size": [128, 128, 3],
  "classes": ["no_damage", "damage"],
  "preprocessing": {
    "resize": [128, 128],
    "scale": 0.00392156862745098
  }
}
```

**Response Fields:**
- `model_name`: Name of the trained model
- `test_auc`: Test AUC score of the model
- `input_size`: Expected input image dimensions [width, height, channels]
- `classes`: List of possible class labels
- `preprocessing`: Preprocessing parameters (resize dimensions and normalization scale)

### 2. POST /inference

Performs image classification on an uploaded image.

**Request:**
```http
POST /inference HTTP/1.1
Host: localhost:5000
Content-Type: multipart/form-data

[image binary data]
```

**Response:**
```json
{
  "prediction": "damage"
}
```

or

```json
{
  "prediction": "no_damage"
}
```

**Request Formats:**
- **Multipart form data:** Send image as `image` field in multipart form
- **Raw binary:** Send image bytes directly in request body

**Response Fields:**
- `prediction`: Classification result, either `"damage"` or `"no_damage"`

---

## Example Requests

### Using cURL

#### 1. Get Model Summary

```bash
curl http://localhost:5000/summary
```

**Pretty-printed output:**
```bash
curl http://localhost:5000/summary | python -m json.tool
```

#### 2. Perform Inference (Multipart Form)

```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  http://localhost:5000/inference
```

**Example with a test image:**
```bash
curl -X POST \
  -F "image=@./data/damage/-93.66109_30.212114.jpeg" \
  http://localhost:5000/inference
```

#### 3. Perform Inference (Raw Binary)

```bash
curl -X POST \
  --data-binary @/path/to/your/image.jpg \
  -H "Content-Type: application/octet-stream" \
  http://localhost:5000/inference
```

**Example:**
```bash
curl -X POST \
  --data-binary @./data/no_damage/-95.06212_29.829257000000002.jpeg \
  -H "Content-Type: application/octet-stream" \
  http://localhost:5000/inference
```

### Using Python requests Library

#### 1. Get Model Summary

```python
import requests

response = requests.get("http://localhost:5000/summary")
print(response.json())
```

#### 2. Perform Inference (Multipart Form)

```python
import requests

url = "http://localhost:5000/inference"
with open("path/to/image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    print(response.json())
```

#### 3. Perform Inference (Raw Binary)

```python
import requests

url = "http://localhost:5000/inference"
with open("path/to/image.jpg", "rb") as f:
    image_bytes = f.read()
    response = requests.post(url, data=image_bytes)
    print(response.json())
```

### Using Python http.client

```python
import http.client
import json

# Get summary
conn = http.client.HTTPConnection("localhost", 5000)
conn.request("GET", "/summary")
response = conn.getresponse()
data = json.loads(response.read())
print(data)

# Perform inference
with open("path/to/image.jpg", "rb") as f:
    image_data = f.read()
    conn.request("POST", "/inference", image_data)
    response = conn.getresponse()
    result = json.loads(response.read())
    print(result)
```

---

## Running the Server

### Step 1: Start the Server

```bash
docker-compose up -d
```

### Step 2: Verify Server is Running

```bash
# Check container status
docker-compose ps

# Test the summary endpoint
curl http://localhost:5000/summary
```

### Step 3: Test Inference

```bash
# Test with a sample image
curl -X POST \
  -F "image=@./data/damage/-93.66109_30.212114.jpeg" \
  http://localhost:5000/inference
```

Expected output:
```json
{"prediction": "damage"}
```

---

## Testing with the Grader

The project includes an automated grader script to test the inference server.

### Running the Grader

1. **Start the inference server:**
   ```bash
   docker-compose up -d
   ```

2. **Run the grader script:**
   ```bash
   ./start_grader.sh
   ```

### What the Grader Tests

1. **GET /summary endpoint:**
   - Verifies the endpoint accepts GET requests
   - Validates the JSON response format
   - Checks for required fields

2. **POST /inference endpoint:**
   - Tests with multiple images from both classes
   - Verifies the response format
   - Checks prediction accuracy
   - Calculates overall accuracy

### Expected Grader Output

```
**** STARTING GRADING ****

GET /summary format correct; response: {...}
Starting full POST test suite...
POST /inference format correct for input /data/damage/... AND prediction was correct!
POST /inference format correct for input /data/damage/... AND prediction was correct!
...
Final results:
Total correct: 6
Total Inferences: 6
Accuracy: 1.0
```

---

## Troubleshooting

### Server Won't Start

**Check if port 5000 is already in use:**
```bash
lsof -i :5000
# or
netstat -an | grep 5000
```

**Stop any conflicting services or change the port in docker-compose.yml**

### Container Exits Immediately

**View container logs:**
```bash
docker-compose logs inference-server
```

**Common issues:**
- Missing model file in `artifacts/` directory
- Missing preprocessing.json file
- Insufficient memory

### Cannot Connect to Server

**Verify container is running:**
```bash
docker-compose ps
```

**Check if server is accessible:**
```bash
curl http://localhost:5000/summary
```

**If using Docker on remote host, ensure port forwarding is configured**

### Model Loading Errors

**Verify artifacts directory structure:**
```bash
ls -la artifacts/
```

Required files:
- `best_model.keras`
- `preprocessing.json`
- `model_card.json`

### Image Format Issues

The server accepts common image formats (JPEG, PNG, etc.). If you encounter errors:
- Ensure the image file is not corrupted
- Check file permissions
- Verify the image can be opened with PIL/Pillow

### Rebuilding the Image

If you've made changes to the code:

```bash
# Stop the server
docker-compose down

# Rebuild the image
docker build -t ssenthil99/inference-server:latest .

# Or rebuild without cache
docker build --no-cache -t ssenthil99/inference-server:latest .

# Start the server
docker-compose up -d
```

### Viewing Detailed Logs

```bash
# Follow logs in real-time
docker-compose logs -f inference-server

# View last 100 lines
docker-compose logs --tail=100 inference-server
```

---

## Project Structure

```
COE_Project2/
├── artifacts/
│   ├── best_model.keras          # Trained model
│   ├── model_card.json            # Model metadata
│   └── preprocessing.json         # Preprocessing config
├── data/                          # Test images
│   ├── damage/
│   └── no_damage/
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker image definition
├── inference_server.py            # Flask application
├── grader.py                      # Automated testing script
├── start_grader.sh                # Grader execution script
└── README.md                      # This file
```

---

## Additional Notes

- The server runs on port 5000 by default
- The model expects 128x128 RGB images
- Images are automatically resized and normalized
- The server handles both multipart form data and raw binary image uploads
- Container automatically restarts unless manually stopped
- Model is loaded once at startup for optimal performance

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review container logs: `docker-compose logs inference-server`
3. Verify all required files are present in the `artifacts/` directory

---

**Last Updated:** November 13th, 2025
