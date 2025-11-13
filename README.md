# Inference Server - Docker Deployment

## Architecture
This Docker image is built for **x86 architecture** and was built on the class VM to ensure compatibility.

## Starting and Stopping the Inference Server

### Start the Inference Server
```bash
docker-compose up -d
```

The `-d` flag runs the container in detached mode (in the background).

### Stop the Inference Server
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Check Status
```bash
docker-compose ps
```

### Restart the Server
```bash
docker-compose restart
```

## Running the Grader

1. **Start your inference server first:**
   ```bash
   docker-compose up -d
   ```

2. **Run the grader:**
   ```bash
   ./start_grader.sh
   ```

   Or manually:
   ```bash
   docker run -it --rm \
     -v $(pwd)/data:/data \
     -v $(pwd)/grader.py:/grader.py \
     -v $(pwd)/project3-results:/results \
     --entrypoint=python \
     jstubbs/coe379l /grader.py
   ```

3. **The grader will:**
   - Test the `/summary` endpoint (GET request)
   - Test the `/inference` endpoint (POST requests)
   - Calculate accuracy on test images
   - Display results

## Making Requests to the Inference Server

### 1. Health Check (GET)
```bash
curl http://localhost:5000/health
```

**Expected Response:**
```json
{"status": "ok"}
```

### 2. Model Summary (GET)
```bash
curl http://localhost:5000/summary
```

**Expected Response:**
```json
{
  "model_name": "your_model_name",
  "test_auc": 0.95,
  "input_size": [128, 128, 3],
  "classes": ["no_damage", "damage"],
  "preprocessing": {
    "resize": [128, 128],
    "scale": 0.00392156862745098
  }
}
```

### 3. Inference (POST) - Using curl
```bash
curl -X POST http://localhost:5000/inference \
  -F "image=@/path/to/your/image.jpg"
```

**Expected Response:**
```json
{"prediction": "damage"}
```
or
```json
{"prediction": "no_damage"}
```

### 4. Inference (POST) - Using Python
```python
import requests

url = "http://localhost:5000/inference"
with open("path/to/image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    print(response.json())
```

### 5. Inference (POST) - Using Python requests with raw bytes
```python
import requests

url = "http://localhost:5000/inference"
with open("path/to/image.jpg", "rb") as f:
    image_bytes = f.read()
    response = requests.post(url, data=image_bytes)
    print(response.json())
```

## Docker Image Information

- **Image:** `ssenthil99/inference-server:latest`
- **Architecture:** x86_64 (built on class VM)
- **Port:** 5000
- **Available on Docker Hub:** https://hub.docker.com/r/ssenthil99/inference-server

## Pulling the Image

If you need to pull the image on a different machine:
```bash
docker pull ssenthil99/inference-server:latest
```

## Troubleshooting

### Check if the server is running:
```bash
docker-compose ps
curl http://localhost:5000/health
```

### View server logs:
```bash
docker-compose logs inference-server
```

### Rebuild and restart:
```bash
docker-compose down
docker-compose pull
docker-compose up -d
```

