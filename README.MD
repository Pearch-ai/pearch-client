# Pearch Client

Python client library for the Pearch API.

## Quick Setup

### 1. Clone and Setup Environment
```bash
# Activate your conda environment
conda activate chat

# Install the package in editable mode (recommended)
pip install -e .
```

### 2. Environment Variables
```bash
# Copy example environment file
cp env.example .env

# Edit .env with your API credentials
```

### 3. Run Scripts
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

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development setup instructions.

## Installation Methods

### Method 1: Editable Installation (Recommended)
```bash
pip install -e .
```
- ✅ No PYTHONPATH manipulation needed
- ✅ Works from any directory  
- ✅ Standard Python packaging
- ✅ Perfect IDE support

### Method 2: Module Execution
```bash
python -m pearch.api_test
```
- ✅ Built-in Python feature
- ✅ No installation required
- ✅ Run from project root

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
