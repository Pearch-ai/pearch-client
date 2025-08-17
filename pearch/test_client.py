#!/usr/bin/env python3
"""Parallel API endpoint testing script.
Tests multiple endpoints concurrently with configurable host and API key.
"""
import logging
import os
import pytest
import json
from typing import Any


from pearch.client import AsyncPearchClient
from pearch.schema import (
    V1FindMatchingJobsRequest,
    V1ProfileRequest,
    V1SearchRequest,
    V1UpsertJobsRequest,
    V2SearchRequest,
    V2SearchCompanyLeadsRequest,
    Job,
)

logger = logging.getLogger(__name__)


def generate_curl_command(client_method: str, request: Any) -> str:
    """Generate a curl command for reproducing the request."""
    # Map client methods to API endpoints and HTTP methods
    method_mapping = {
        "find_matching_jobs": ("POST", "v1/find_matching_jobs"),
        "get_profile": ("GET", "v1/profile"),
        "search_v1": ("GET", "v1/search"),
        "upsert_jobs": ("POST", "v1/upsert_jobs"),
        "search": ("POST", "v2/search"),
        "search_company_leads": ("POST", "v2/search_company_leads"),
    }

    if client_method not in method_mapping:
        return f"# Unable to generate curl for {client_method}"

    http_method, endpoint = method_mapping[client_method]
    base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"
    api_key = os.getenv("PEARCH_API_KEY")
    token = os.getenv("PEARCH_TEST_KEY")
    url = f"{base_url.rstrip('/')}/{endpoint}"

    curl_parts = ["curl", "-X", http_method]
    curl_parts.extend(["-H", f"'Authorization: Bearer {api_key}'"])
    curl_parts.extend(["-H", "'Content-Type: application/json'"])

    if token:
        curl_parts.extend(["-H", f"'X-Test-Secret: {token}'"])

    if http_method == "POST" and hasattr(request, "model_dump"):
        request_json = json.dumps(request.model_dump(exclude_none=True))
        curl_parts.extend(["-d", f"'{request_json}'"])
    elif http_method == "GET" and hasattr(request, "model_dump"):
        params = request.model_dump(exclude_none=True)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            url += f"?{query_string}"

    curl_parts.append(f"'{url}'")
    logger.info(" ".join(curl_parts))


@pytest.mark.asyncio
async def test_find_matching_jobs():
    request = V1FindMatchingJobsRequest(
        profile={
            "description": "I'm a Senior Silicon Design Engineer with strong background in computer engineering, hands-on silicon design experience, and deep understanding of the semiconductor industry."
        },
        limit=2,
    )
    generate_curl_command("find_matching_jobs", request)
    response = await AsyncPearchClient().find_matching_jobs(request)
    assert response.jobs
    assert any(job.job_id for job in response.jobs)


@pytest.mark.asyncio
async def test_profile():
    request = V1ProfileRequest(docid="vslaykovsky", show_emails=True)
    generate_curl_command("get_profile", request)
    response = await AsyncPearchClient().get_profile(request)
    assert "vlad" in response.profile.first_name.lower()


@pytest.mark.asyncio
async def test_v1_fast_search():
    request = V1SearchRequest(query="software engineer", limit=100, type="fast")
    generate_curl_command("search_v1", request)
    response = await AsyncPearchClient().search_v1(request)
    assert any(result.linkedin_slug for result in response)


@pytest.mark.asyncio
async def test_upsert_jobs():
    request = V1UpsertJobsRequest(
        jobs=[
            Job(
                job_id="1",
                job_description="software engineer in test for la ai startup",
            )
        ]
    )
    generate_curl_command("upsert_jobs", request)
    response = await AsyncPearchClient().upsert_jobs(request)
    assert "success" in response.status.lower()


@pytest.mark.asyncio
async def test_v2_pro_search():
    request = V2SearchRequest(
        query="Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree",
        limit=2,
    )
    generate_curl_command("search", request)
    response = await AsyncPearchClient().search(request)
    assert any(result.profile.linkedin_slug for result in response.search_results)


@pytest.mark.asyncio
async def test_v2_pro_search_narrow():
    request = V2SearchRequest(query="employees of collectly", limit=2)
    generate_curl_command("search", request)
    response = await AsyncPearchClient().search(request)
    assert response.search_results[0].profile.linkedin_slug is not None


@pytest.mark.asyncio
async def test_search_company_leads():
    request = V2SearchCompanyLeadsRequest(
        company_query="ats companies",
        lead_query="c-levels and founders",
        outreach_message_instruction="<300 characters, email style, casual",
        limit=12,
        leads_limit=2,
        show_emails=True,
    )
    generate_curl_command("search_company_leads", request)
    response = await AsyncPearchClient().search_company_leads(request)
    assert any(
        lead.profile.linkedin_slug
        for company_result in response.search_results
        for lead in (company_result.leads or [])
    )
