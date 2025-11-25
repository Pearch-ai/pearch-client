# Pearch Client

Official Python client for the Pearch.AI API.

Get your PEARCH_API_KEY from https://pearch.ai and add it to your `.env` file or set it another way.

## Requirements

- python >=3.8

## Installation

Install the latest version of the library:
```
pip install git+https://github.com/Pearch-ai/pearch-client
```

Alternatively, install it in editable mode:
```
pip install -e git+https://github.com/Pearch-ai/pearch-client#egg=pearch-client
```


## Usage

Check out API documentation: https://apidocs.pearch.ai/reference/post_v2-search

### Async client (/v2/search)

```python
from pearch.client import AsyncPearchClient
from pearch.schema import V2SearchRequest

QUERY = "Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree"

client = AsyncPearchClient()
results = await client.search(V2SearchRequest(query=QUERY))
print(results.model_dump_json()[:1000])
```

```python
# Or use with context manager for automatic cleanup
async with AsyncPearchClient() as client:
    results = await client.search(V2SearchRequest(query=QUERY))
print(results.model_dump_json()[:1000])
```

Output:
```bash
INFO - pearch.client - client.py:471 - Executing async v2 search with query: Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree
INFO - httpx - _client.py:1740 - HTTP Request: POST https://api.pearch.ai/v2/search "HTTP/1.1 200 OK"
{"uuid":"fd975932-4e79-4ce7-abb6-9f8b7f796f82","query":"Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree","user":"dev@dev.co","created_at":1755424260.2801638,"duration":6.51,"status":"Done","total_estimate":80,"credits_remaining":42,"search_results":[{"docid":"po-lung-chen-59a07127","profile":{"docid":"po-lung-chen-59a07127","linkedin_slug":"po-lung-chen-59a07127","first_name":"Po-lung","middle_name":null,"last_name":"Chen","picture_url":"https://media.licdn.com/dms/image/v2/C5603AQFjOXfIJWTpGw/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1617665624267?e=2147483647&v=beta&t=1eYMjqT6cuU4vLAvTxsnXU2bJdW4bkn9Mj7JJJedHkE","title":"Software Engineering Manager at Instagram","summary":"A full-stack Software Engineering tech-leading and managing role in Facebook. I'm passionate about observing and facing various flows i
```

## Sync client (/v2/search)

```python
from pearch.client import PearchClient
from pearch.schema import V2SearchRequest

QUERY = "Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree"

client = PearchClient()
results = client.search(V2SearchRequest(query=QUERY))
```

```python
# Or use with context manager for automatic cleanup
with PearchClient() as client:
    results = client.search(V2SearchRequest(query=QUERY))
```

## Custom Filters

Use structured `CustomFilters` for type-safe filtering with IDE autocomplete:

```python
from pearch.client import PearchClient
from pearch import V2SearchRequest, CustomFilters

with PearchClient() as client:
    # Create structured filters with type hints and validation
    filters = CustomFilters(
        locations=["San Francisco", "New York"],
        industries=["health", "pharma"],
        titles=["software engineer", "data scientist"],
        min_total_experience_years=5,
        max_total_experience_years=15,
        studied_at_top_universities=True,
        has_startup_experience=True,
        degrees=["bachelor", "master"],
        specialization_categories=["Computer Science & IT", "Engineering"],
        min_linkedin_followers=500,
        gender="male"
    )
    
    results = client.search(V2SearchRequest(
        query="senior engineer with ML experience",
        custom_filters=filters,
        limit=20
    ))
```

Available filter fields:
- **Array filters**: `locations`, `languages`, `titles`, `industries`, `companies`, `universities`, `keywords`, `degrees`, `specialization_categories`
- **Numeric ranges**: `min/max_linkedin_followers`, `min/max_total_experience_years`, `min/max_estimated_age`, `min/max_current_experience_years`
- **Boolean filters**: `studied_at_top_universities`, `has_startup_experience`, `has_saas_experience`, `has_b2b_experience`, `has_b2c_experience`
- **Exact match**: `first_name`, `middle_name`, `last_name`, `gender`

Alternatively, use a plain dictionary for backward compatibility:

```python
results = client.search(V2SearchRequest(
    query="software engineer",
    custom_filters={
        "locations": ["San Francisco"],
        "min_total_experience_years": 3
    }
))
```

## Job Management

The Pearch client provides comprehensive job management capabilities:

- **Upload/Update Jobs** (`/v1/upsert_jobs`) - Add or update job listings in your index
- **List Jobs** (`/v1/list_jobs`) - Retrieve all jobs with pagination and metadata  
- **Delete Jobs** (`/v1/delete_jobs`) - Remove specific jobs by ID
- **Find Matching Jobs** (`/v1/find_matching_jobs`) - Discover relevant jobs for candidate profiles

### Index jobs (/v1/upsert_jobs)

```python
from pearch.client import PearchClient
from pearch.schema import V1UpsertJobsRequest, V1FindMatchingJobsRequest, V1ListJobsRequest, V1DeleteJobsRequest, Job

# Upload or update job listings
with PearchClient() as client:
    response = client.upsert_jobs(V1UpsertJobsRequest(
        jobs=[
            Job(
                job_id="1",
                job_description="software engineer in test for la ai startup",
            )
        ]
    ))
    print(response.uuid)
    print(response.status)
    print(f"Upsert status: {response.model_dump_json()}")
```
Output:
```bash
INFO - pearch.client - client.py:270 - Upserting 1 jobs (replace=False)
INFO - httpx - _client.py:1025 - HTTP Request: POST https://api.pearch.ai/v1/upsert_jobs "HTTP/1.1 200 OK"
e7d685cd-e669-4b1c-b4e4-276ed607d735
success
Upsert status: {"uuid":"e7d685cd-e669-4b1c-b4e4-276ed607d735","status":"success","processed_count":1}
```

### Find relevant jobs (/v1/find_matching_jobs)
```
# Find jobs matching a candidate profile  
with PearchClient() as client:
    response = client.find_matching_jobs(V1FindMatchingJobsRequest(
        profile={
            "description": "I'm a Senior Silicon Design Engineer with strong background in computer engineering, hands-on silicon design experience, and deep understanding of the semiconductor industry."
        },
        limit=2,
    ))
    for job in response.jobs:
        print(job.job_id, job.model_dump_json()[:100])
```
Output:
```bash
INFO - pearch.client - client.py:295 - Finding matching jobs for profile (limit=2)
INFO - httpx - _client.py:1025 - HTTP Request: POST https://api.pearch.ai/v1/find_matching_jobs "HTTP/1.1 200 OK"
83 {"job_id":"83","job_description":"We are seeking a Senior Silicon Design Enginee
84 {"job_id":"84","job_description":"Join a leading enterprise in the medical imagi
```

### List jobs (/v1/list_jobs)

```python
from pearch.client import PearchClient
from pearch.schema import V1ListJobsRequest

# List all jobs in your index
with PearchClient() as client:
    response = client.list_jobs(V1ListJobsRequest(limit=10))
    print(f"Total jobs: {response.total_count}")
    print(f"Returned jobs: {response.returned_count}")
    
    for job in response.jobs:
        print(f"Job ID: {job.job_id}")
        print(f"Description: {job.job_description}")
        print(f"Created: {job.created_at}")
        print("---")
```

Output:
```bash
INFO - pearch.client - client.py:344 - Listing jobs with limit: 10
INFO - httpx - _client.py:1025 - HTTP Request: GET https://api.pearch.ai/v1/list_jobs?limit=10 "HTTP/1.1 200 OK"
Total jobs: 25
Returned jobs: 10
Job ID: test_job_1
Description: Senior Software Engineer for AI startup in LA
Created: 2024-01-15T10:30:45
---
Job ID: test_job_2
Description: Frontend Developer with React experience
Created: 2024-01-15T09:15:22
---
```

### Delete jobs (/v1/delete_jobs)

```python
from pearch.client import PearchClient
from pearch.schema import V1DeleteJobsRequest

# Delete specific jobs by their IDs
with PearchClient() as client:
    response = client.delete_jobs(V1DeleteJobsRequest(
        job_ids=["test_job_1", "test_job_2"]
    ))
    print(f"Status: {response.status}")
    print(f"Deleted count: {response.deleted_count}")
    if response.errors:
        print(f"Errors: {response.errors}")
```

Output:
```bash
INFO - pearch.client - client.py:368 - Deleting 2 jobs
INFO - httpx - _client.py:1025 - HTTP Request: POST https://api.pearch.ai/v1/delete_jobs "HTTP/1.1 200 OK"
Status: success
Deleted count: 2
```
 

### Enrich profile (/v1/profile)

```python
from pearch.client import PearchClient
from pearch.schema import V1ProfileRequest

# Retrieve detailed profile information
with PearchClient() as client:
    request = V1ProfileRequest(
        docid="vslaykovsky", 
        show_emails=True,
        with_profile=True
    )
    response = client.get_profile(request)
    print(f"Profile: {response.model_dump_json()[:1000]}")
```

Available parameters:
- `docid` (required): LinkedIn profile ID
- `high_freshness` (optional): Enable real-time profile updates (+2 credits)
- `show_emails` (optional): Include email addresses if available (+2 credits if emails found)
- `show_phone_numbers` (optional): Include phone numbers if available (+14 credits if phone numbers found)
- `with_profile` (optional): Include full profile data in response (+1 credit)

Output:
```bash
INFO - pearch.client - client.py:321 - Retrieving profile for docid: vslaykovsky
INFO - httpx - _client.py:1025 - HTTP Request: GET https://api.pearch.ai/v1/profile?docid=vslaykovsky&high_freshness=false&show_emails=true&show_phone_numbers=false&with_profile=true "HTTP/1.1 200 OK"
Profile: {"uuid":"1f3cca0a-06b2-4679-95eb-d9b1075a2764","profile":{"docid":"vslaykovsky","linkedin_slug":"vslaykovsky","first_name":"Vlad","middle_name":null,"last_name":"Sly","picture_url":"https://media.licdn.com/dms/image/v2/C4D03AQE9ZowG0kPs6A/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1538525794823?e=2147483647&v=beta&t=hZOiV9Po50zLV9blRTbra2IfeXvSdkeuJyhFJ4rmr9k","title":"Machine Learning Researcher + Kaggle Grandmaster, top10","summary":"Machine Learning Researcher with extensive experience in industry. Fluent in Python, multiple years of experience building scalable backend systems in C/C++, Java and PHP.","gender":null,"is_decision_maker":1.0,"languages":[{"language":"English","proficiency":"C1"},{"language":"Russian","proficiency":"C2"}],"inferred_languages":[],"location":"Bristol, England, United Kingdom","is_top_universities":false,"is_opentowork":null,"is_hiring":null,"total_experience_years":22.0,"estimated_age":43.0,"expertise":["machine learning","
```

## OpenAI-Compatible Chat Completions (/v1/chat/completions)

The Pearch API provides an OpenAI-compatible chat completions endpoint that supports streaming search results. This endpoint accepts the standard OpenAI format and returns search results in a streaming format.

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.pearch.ai/",
    api_key="your-api-key"
)

# Streaming search
stream = client.chat.completions.create(
    model="pearch",
    stream=True,
    messages=[{"role": "user", "content": "software engineers at Google with 10 years of experience"}],
    extra_body={
        "limit": 5,
        "type": "fast"
    }
)

for chunk in stream:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)
        if chunk.choices[0].finish_reason is not None:
            print(f"\nFinished: {chunk.choices[0].finish_reason}")
```

### Streaming Options

The endpoint supports several streaming modes via `extra_body`:

- **`stream_profiles`**: Control how profiles are streamed
  - `"full"`: Stream each profile as it's found
  - `"batch"`: Stream profiles in batches
  - `None`: Don't stream profiles (default)

- **`profiles_batch_size`**: Number of profiles per batch (default: 10)

- **`final_result`**: Include full result in final chunk (default: True)

### Advanced Parameters

All search parameters can be passed via `extra_body`:

```python
stream = client.chat.completions.create(
    model="pearch",
    stream=True,
    messages=[{"role": "user", "content": "software engineers in California"}],
    extra_body={
        "limit": 10,
        "type": "pro",  # or "fast"
        "stream_profiles": "batch",
        "profiles_batch_size": 5,
        "final_result": True,
        "insights": True,
        "high_freshness": True,
        "show_emails": True,
        "show_phone_numbers": True,
        "require_emails": True,
        "require_phone_numbers": True,
        "profile_scoring": True,
        "strict_filters": False,
        "custom_filters": {}
    }
)
```

### Thread Continuation

You can continue a search thread by passing `thread_id`:

```python
# First request
stream1 = client.chat.completions.create(
    model="pearch",
    stream=True,
    messages=[{"role": "user", "content": "software engineers at Google"}],
    extra_body={"limit": 5, "type": "fast"}
)

# Extract thread_id from final chunk
thread_id = None
for chunk in stream1:
    if chunk.choices and chunk.choices[0].finish_reason is not None:
        # Extract thread_id from chunk summary if available
        break

# Continue with same thread
stream2 = client.chat.completions.create(
    model="pearch",
    stream=True,
    messages=[{"role": "user", "content": "who are at least 30 years old"}],
    extra_body={"limit": 5, "thread_id": thread_id, "type": "fast"}
)
```

### Note on Streaming

**Important**: The `/v1/chat/completions` endpoint always returns a streaming response (SSE format), even when `stream=False` is specified. You should always use `stream=True` and process the response as a stream.

## Testing 

These tests hit pearch API, so they require PEARCH_API_KEY and may incur costs. See `pearch/test_client.py`. 
Available in editable installation. 

```
pytest
```

 
# Development

- Only backward compatible changes are allowed for existing endpoints.
  - Backward compatible: add new fields; copy fields to new objects but keep originals.
  - Incompatible: change field types; move and remove fields.
- Breaking changes require a new endpoint version.
- Update both `pearch-openapi.yaml` and `pearch/schema.py` for any changes.