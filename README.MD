# Pearch Client

Python client library for the Pearch API.

## Quick Setup

### 1. Clone and Setup Environment
```bash
# Install the package in editable mode (recommended)
pip install -e .
```

### 2. Environment Variables
```bash
# Copy example environment file
cp env.example .env

# Edit .env with your API credentials
```

### 3. [Optional] Run Scripts
```bash
# Run the API test script
python pearch/api_test.py

# Or run as a module
python -m pearch.api_test
```

## Usage

```python
from pearch.client import AsyncPearchClient
from pearch.schema import V1SearchRequest

# Initialize client
client = AsyncPearchClient()

# Use the client
request = V1SearchRequest(query="software engineer")
results = await client.search(request)
```
 

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
