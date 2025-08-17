# Pearch Client

Python client library for the Pearch.AI API. 
To use the library you need your PEARCH_API_KEY, get it at https://pearch.ai

Update PEARCH_API_KEY in your `.env` file or using any alternative method. 


## Requirements

- python >=3.8

## Installation

Install the latest version of the library:
```
pip install git+https://github.com/Pearch-ai/pearch-client
```

Alternatively, install it in editable mode:
```
pip install -e git+https://github.com/Pearch-ai/pearch-client
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

## Testing 

These tests hit pearch API, so they require PEARCH_API_KEY and may incur costs. See `pearch/test_client.py`

```
pytest --pyargs pearch-client
```

 
# Development

- Only backward compatible changes are allowed for existing endpoints.
  - Backward compatible: add new fields; copy fields to new objects but keep originals.
  - Incompatible: change field types; move and remove fields.
- Breaking changes require a new endpoint version.
- Update both `pearch-openapi.yaml` and `pearch/schema.py` for any changes.