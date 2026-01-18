#!/usr/bin/env python3
"""Parallel API endpoint testing script.
Tests multiple endpoints concurrently with configurable host and API key.
"""
import asyncio
import logging
import os
import pytest
import json
from typing import Any
import time
import uuid

from openai import OpenAI

from pearch.client import AsyncPearchClient
from pearch.schema import (
    V1FindMatchingJobsRequest,
    V1FindMatchingJobsResponse,
    V1ProfileRequest,
    V1ProfileResponse,
    V1ApiCallHistoryRequest,
    V1SearchRequest,
    V1UpsertJobsRequest,
    V1UpsertJobsResponse,
    V1ListJobsRequest,
    V1ListJobsResponse,
    V1DeleteJobsRequest,
    V1DeleteJobsResponse,
    V1UserResponse,
    V2SearchCompanyLeadsResponse,
    V2SearchRequest,
    V2SearchCompanyLeadsRequest,
    Job,
    V2SearchResponse,
    V2SearchStatusResponse,
)

logger = logging.getLogger(__name__)


def generate_curl_command(client_method: str, request: Any) -> str:
    """Generate a curl command for reproducing the request."""
    # Map client methods to API endpoints and HTTP methods
    method_mapping = {
        "find_matching_jobs": ("POST", "v1/find_matching_jobs"),
        "get_profile": ("GET", "v1/profile"),
        "api_call_history": ("GET", "v1/api_call_history"),
        "get_user": ("GET", "v1/user"),
        "search_v1": ("GET", "v1/search"),
        "upsert_jobs": ("POST", "v1/upsert_jobs"),
        "list_jobs": ("GET", "v1/list_jobs"),
        "delete_jobs": ("POST", "v1/delete_jobs"),
        "search": ("POST", "v2/search"),
        "search_company_leads": ("POST", "v2/search_company_leads"),
        "search_submit": ("POST", "v2/search/submit"),
        "get_search_status": ("GET", "v2/search/status"),
    }

    if client_method not in method_mapping:
        return f"# Unable to generate curl for {client_method}"

    http_method, endpoint = method_mapping[client_method]
    base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"
    api_key = os.getenv("PEARCH_API_KEY")
    token = os.getenv("PEARCH_TEST_KEY")
    
    if client_method == "get_search_status":
        url = f"{base_url.rstrip('/')}/{endpoint}/{request['task_id']}"
    else:
        url = f"{base_url.rstrip('/')}/{endpoint}"

    curl_parts = ["curl", "-X", http_method]
    curl_parts.extend(["-H", f"'Authorization: Bearer {api_key}'"])
    curl_parts.extend(["-H", "'Content-Type: application/json'"])

    if token:
        curl_parts.extend(["-H", f"'X-Test-Secret: {token}'"])

    if http_method == "POST" and hasattr(request, "model_dump"):
        request_json = json.dumps(request.model_dump(exclude_none=True))
        curl_parts.extend(["-d", f"'{request_json}'"])
    elif http_method == "GET" and request is not None and hasattr(request, "model_dump"):
        params = request.model_dump(exclude_none=True)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            url += f"?{query_string}"

    curl_parts.append(f"'{url}'")
    logger.info(" ".join(curl_parts))


def generate_threads_runs_stream_curl(thread_id: str, assistant_id: str, input_data: dict, config: dict, stream_mode: list, stream_resumable: bool, on_disconnect: str) -> str:
    """Generate a curl command for threads/runs/stream endpoint."""
    base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"
    api_key = os.getenv("PEARCH_API_KEY")
    token = os.getenv("PEARCH_TEST_KEY")
    
    url = f"{base_url.rstrip('/')}/threads/{thread_id}/runs"
    
    query_params = [f"assistant_id={assistant_id}"]
    for mode in stream_mode:
        query_params.append(f"stream_mode={mode}")
    if stream_resumable:
        query_params.append("stream_resumable=true")
    if on_disconnect:
        query_params.append(f"on_disconnect={on_disconnect}")
    
    if query_params:
        url += "?" + "&".join(query_params)
    
    body_data = {
        "input": input_data,
        "config": config
    }
    body_json = json.dumps(body_data, separators=(',', ':'))
    
    curl_parts = ["curl", "-X", "POST", "-N"]
    curl_parts.extend(["-H", f"'Authorization: Bearer {api_key}'"])
    curl_parts.extend(["-H", "'Content-Type: application/json'"])
    
    if token:
        curl_parts.extend(["-H", f"'X-Test-Secret: {token}'"])
    
    curl_parts.extend(["-d", f"'{body_json}'"])
    curl_parts.append(f"'{url}'")
    
    curl_command = " ".join(curl_parts)
    logger.info(curl_command)
    return curl_command


async def get_credits():
    response: V1UserResponse = await AsyncPearchClient().get_user()
    return response.credits_remaining

@pytest.mark.asyncio
async def test_find_matching_jobs():
    credits1 = await get_credits()
    request = V1FindMatchingJobsRequest(
        profile={
            "description": "I'm a Senior Silicon Design Engineer with strong background in computer engineering, hands-on silicon design experience, and deep understanding of the semiconductor industry."
        },
        limit=2,
    )
    generate_curl_command("find_matching_jobs", request)
    response: V1FindMatchingJobsResponse = await AsyncPearchClient().find_matching_jobs(request)
    assert response.jobs
    assert any(job.job_id for job in response.jobs)
    assert response.credits_used == len(response.jobs) * 1
    credits2 = await get_credits()
    assert credits1 - credits2 == response.credits_used, "Credits check failed"

@pytest.mark.asyncio
async def test_profile():
    credits1 = await get_credits()
    request = V1ProfileRequest(docid="vslaykovsky", show_emails=True, show_phone_numbers=True)
    generate_curl_command("get_profile", request)
    response: V1ProfileResponse = await AsyncPearchClient().get_profile(request)
    assert response.credits_used == 2 * (0 if not response.profile.get_all_emails() else 1) + 14 * (0 if not response.profile.all_phone_numbers() else 1)
    credits2 = await get_credits()
    assert credits1 - credits2 == response.credits_used, "Credits check failed"

    request = V1ProfileRequest(docid="victorsunden", show_emails=True, high_freshness=True, show_phone_numbers=True, with_profile=True)
    generate_curl_command("get_profile", request)
    response: V1ProfileResponse = await AsyncPearchClient().get_profile(request)
    assert "victor" in response.profile.first_name.lower()
    assert response.credits_used == 2 + 2 * (0 if not response.profile.get_all_emails() else 1) + 14 * (0 if not response.profile.all_phone_numbers() else 1) + 1
    credits3 = await get_credits()
    assert credits2 - credits3 == response.credits_used, "Credits check failed"

@pytest.mark.asyncio
async def test_v1_fast_search():
    credits1 = await get_credits()
    request = V1SearchRequest(query="software engineer", limit=100, type="fast")
    generate_curl_command("search_v1", request)
    response = await AsyncPearchClient().search_v1(request)
    assert any(result.linkedin_slug for result in response)
    credits2 = await get_credits()
    assert credits1 - credits2 == len(response) * 1, "Credits check failed"

@pytest.mark.asyncio
async def test_upsert_jobs():
    credits1 = await get_credits()
    
    # Upsert 3 jobs
    request = V1UpsertJobsRequest(
        jobs=[
            Job(
                job_id="test_job_1",
                job_description="Senior Software Engineer for AI startup in LA",
            ),
            Job(
                job_id="test_job_2", 
                job_description="Frontend Developer with React experience",
            ),
            Job(
                job_id="test_job_3",
                job_description="DevOps Engineer with AWS and Kubernetes skills",
            )
        ]
    )
    generate_curl_command("upsert_jobs", request)
    response: V1UpsertJobsResponse = await AsyncPearchClient().upsert_jobs(request)
    assert response.credits_used == 3
    assert "success" in response.status.lower()
    credits2 = await get_credits()
    assert credits1 - credits2 == response.credits_used, "Credits check failed"

    # List jobs to confirm all 3 jobs were created
    list_request = V1ListJobsRequest(limit=100)
    generate_curl_command("list_jobs", list_request)
    list_response: V1ListJobsResponse = await AsyncPearchClient().list_jobs(list_request)
    assert "success" in list_response.status.lower()
    
    # Check that our test jobs exist
    test_job_ids = {"test_job_1", "test_job_2", "test_job_3"}
    found_job_ids = {job.job_id for job in list_response.jobs if job.job_id in test_job_ids}
    assert len(found_job_ids) == 3, f"Expected 3 test jobs, found: {found_job_ids}"
    assert found_job_ids == test_job_ids, "All 3 test jobs should be present"
    
    # Get initial count for comparison after deletion
    initial_total_count = list_response.total_count
    
    # Delete 2 jobs: one with correct ID and one with incorrect ID
    delete_request = V1DeleteJobsRequest(job_ids=["test_job_1", "nonexistent_job_id"])
    generate_curl_command("delete_jobs", delete_request)
    delete_response: V1DeleteJobsResponse = await AsyncPearchClient().delete_jobs(delete_request)
    assert "success" in delete_response.status.lower()
    assert delete_response.deleted_count == 1, "Only 1 job should be deleted (the existing one)"
    
    # List jobs again to confirm only test_job_1 was deleted
    list_request2 = V1ListJobsRequest(limit=100)
    generate_curl_command("list_jobs", list_request2)
    list_response2: V1ListJobsResponse = await AsyncPearchClient().list_jobs(list_request2)
    assert "success" in list_response2.status.lower()
    
    # Check remaining test jobs
    remaining_test_job_ids = {job.job_id for job in list_response2.jobs if job.job_id in test_job_ids}
    expected_remaining = {"test_job_2", "test_job_3"}
    assert remaining_test_job_ids == expected_remaining, f"Expected jobs {expected_remaining}, found: {remaining_test_job_ids}"
    assert "test_job_1" not in remaining_test_job_ids, "test_job_1 should be deleted"
    
    # Verify total count decreased by 1
    assert list_response2.total_count == initial_total_count - 1, "Total count should decrease by 1"
     

 
def validate_credits(request: V2SearchRequest, response: V2SearchResponse | V2SearchStatusResponse, base_request: V2SearchRequest | None = None):
    """
    Validate credits used in a search response.
    
    Args:
        request: The current request (may have None values when using thread_id)
        response: The search response
        base_request: Optional base request to inherit parameters from when using thread_id
    """
    # When using thread_id, API uses parameters from the original request, not the new one
    # So we should use base_request parameters if available, otherwise use request parameters
    if base_request is not None:
        # Use base_request parameters as primary source when using thread_id
        type_val = base_request.type if base_request.type is not None else (request.type if request.type is not None else "pro")
        insights = base_request.insights if base_request.insights is not None else (request.insights if request.insights is not None else True)
        high_freshness = base_request.high_freshness if base_request.high_freshness is not None else (request.high_freshness if request.high_freshness is not None else False)
        profile_scoring = base_request.profile_scoring if base_request.profile_scoring is not None else (request.profile_scoring if request.profile_scoring is not None else True)
        show_emails = base_request.show_emails if base_request.show_emails is not None else (request.show_emails if request.show_emails is not None else False)
        show_phone_numbers = base_request.show_phone_numbers if base_request.show_phone_numbers is not None else (request.show_phone_numbers if request.show_phone_numbers is not None else False)
        require_emails = base_request.require_emails if base_request.require_emails is not None else (request.require_emails if request.require_emails is not None else False)
        require_phone_numbers = base_request.require_phone_numbers if base_request.require_phone_numbers is not None else (request.require_phone_numbers if request.require_phone_numbers is not None else False)
        require_phones_or_emails = base_request.require_phones_or_emails if base_request.require_phones_or_emails is not None else (request.require_phones_or_emails if request.require_phones_or_emails is not None else False)
    else:
        type_val = request.type if request.type is not None else "pro"
        insights = request.insights if request.insights is not None else True
        high_freshness = request.high_freshness if request.high_freshness is not None else False
        profile_scoring = request.profile_scoring if request.profile_scoring is not None else True
        show_emails = request.show_emails if request.show_emails is not None else False
        show_phone_numbers = request.show_phone_numbers if request.show_phone_numbers is not None else False
        require_emails = request.require_emails if request.require_emails is not None else False
        require_phone_numbers = request.require_phone_numbers if request.require_phone_numbers is not None else False
        require_phones_or_emails = request.require_phones_or_emails if request.require_phones_or_emails is not None else False
    
    expected_credits = 0   
    for result in response.search_results:
        candidate_credits = 0
        profile_id = result.profile.linkedin_slug if result.profile and result.profile.linkedin_slug else "unknown"
        if type_val == "pro":
            candidate_credits += 5
            logger.info(f"{profile_id}: Incremented candidate_credits by 5 for type 'pro', total now {candidate_credits}")
        elif type_val == "fast":
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for type 'fast', total now {candidate_credits}")
        if insights and result.insights:
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for insights, total now {candidate_credits}")
        if high_freshness:
            candidate_credits += 2
            logger.info(f"{profile_id}: Incremented candidate_credits by 2 for high_freshness, total now {candidate_credits}")
        if profile_scoring and result.score is not None:
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for profile_scoring, total now {candidate_credits}")
        if show_emails and result.profile and result.profile.get_all_emails():
            candidate_credits += 2
            logger.info(f"{profile_id}: Incremented candidate_credits by 2 for show_emails, total now {candidate_credits}")
        if show_phone_numbers and result.profile and result.profile.phone_numbers:
            candidate_credits += 14
            logger.info(f"{profile_id}: Incremented candidate_credits by 14 for show_phone_numbers, total now {candidate_credits}")
        if require_emails or require_phone_numbers or require_phones_or_emails:
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for require_emails or require_phone_numbers or require_phones_or_emails, total now {candidate_credits}")
        expected_credits += candidate_credits
        logger.info(f"{profile_id}: Added candidate_credits {candidate_credits} to expected_credits, expected_credits now {expected_credits}")
    assert response.credits_used == expected_credits    



@pytest.mark.asyncio
async def test_v2_fast_search():
    credits1 = await get_credits()
    first_request = V2SearchRequest(
        query="Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree",
        type="fast",
        limit=2,
        show_emails=True,
        show_phone_numbers=True,
        insights=True,
        high_freshness=True,
        profile_scoring=True,
        require_emails=True,
        require_phone_numbers=True,        
    )
    generate_curl_command("search", first_request)
    response: V2SearchResponse = await AsyncPearchClient().search(first_request)
    assert len(response.search_results) == 2
    assert any(result.profile.linkedin_slug for result in response.search_results)
    assert all(len(result.profile.get_all_emails()) > 0 for result in response.search_results)
    assert all(len(result.profile.all_phone_numbers()) > 0 for result in response.search_results)
    validate_credits(first_request, response)
    credits2 = await get_credits()
    assert credits1 - credits2 == response.credits_used, "Credits check failed"

@pytest.mark.asyncio
async def test_v2_pro_search_generic():
    credits1 = await get_credits()
    logger.info(f"Credits1: {credits1}")
    first_request = V2SearchRequest(
        query="Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree",
        limit=2,
        show_emails=True,
        show_phone_numbers=True,
        insights=True,
        high_freshness=True,
        profile_scoring=True,
        require_emails=True,
        require_phone_numbers=True,        
    )
    generate_curl_command("search", first_request)
    response: V2SearchResponse = await AsyncPearchClient().search(first_request)
    assert len(response.search_results) == 2
    assert any(result.profile.linkedin_slug for result in response.search_results)
    assert all(len(result.profile.get_all_emails()) > 0 for result in response.search_results)
    assert all(len(result.profile.all_phone_numbers()) > 0 for result in response.search_results)
    validate_credits(first_request, response)
    credits2 = await get_credits()
    logger.info(f"Credits2: {credits2}")
    assert credits1 - credits2 == response.credits_used, "Credits check failed"

    # "show more"
    second_request = V2SearchRequest(
        limit=4,
        thread_id=response.thread_id,
    )
    generate_curl_command("search", second_request)
    response: V2SearchResponse = await AsyncPearchClient().search(second_request)
    assert len(response.search_results) == 4
    all_results = response.search_results
    new_results = response.search_results[2:4]  # Last 2 are new
    response.search_results = new_results
    validate_credits(second_request, response, base_request=first_request)
    response.search_results = all_results
    credits3 = await get_credits()
    logger.info(f"Credits3: {credits3}")
    assert credits2 - credits3 == response.credits_used, "Credits check failed"

    # follow up query
    third_request = V2SearchRequest(
        thread_id=response.thread_id,
        query="who are at least 30 years old",
        limit=2,
    )
    generate_curl_command("search", third_request)
    response: V2SearchResponse = await AsyncPearchClient().search(third_request)
    assert len(response.search_results) == 2
    validate_credits(third_request, response, base_request=first_request)
    credits4 = await get_credits()
    logger.info(f"Credits4: {credits4}")
    assert credits3 - credits4 == response.credits_used, "Credits check failed"
 
  

def validate_company_leads_credits(request: V2SearchCompanyLeadsRequest, response: V2SearchCompanyLeadsResponse):
    expected_credits = 0
    
    # Count companies and leads in response
    num_companies = len(response.search_results) if response.search_results else 0
    total_leads = 0
    
    if response.search_results:
        for company_result in response.search_results:
            if company_result.leads:
                total_leads += len(company_result.leads)
    
    # Company-level costs
    # company_query: 5 credits per company (required)
    company_credits = num_companies * 5
    expected_credits += company_credits
    logger.info(f"Added {company_credits} credits for {num_companies} companies (5 credits each), total now {expected_credits}")
    
    # company_high_freshness: 2 credits per company (optional)
    if request.company_high_freshness:
        company_freshness_credits = num_companies * 2
        expected_credits += company_freshness_credits
        logger.info(f"Added {company_freshness_credits} credits for company_high_freshness ({num_companies} companies * 2), total now {expected_credits}")
    
    # Lead-level costs (only if we have leads)
    if total_leads > 0:
        # lead_query: 2 credits per lead (optional, only if provided)
        if request.lead_query:
            lead_query_credits = total_leads * 2
            expected_credits += lead_query_credits
            logger.info(f"Added {lead_query_credits} credits for lead_query ({total_leads} leads * 2), total now {expected_credits}")
        
        # outreach_message_instruction: 3 credits per lead (optional, only if provided)
        if request.outreach_message_instruction:
            # Only count leads that actually have outreach messages
            leads_with_outreach = 0
            if response.search_results:
                for company_result in response.search_results:
                    if company_result.leads:
                        for lead in company_result.leads:
                            if lead.outreach_message:
                                leads_with_outreach += 1
            outreach_credits = leads_with_outreach * 3
            expected_credits += outreach_credits
            logger.info(f"Added {outreach_credits} credits for outreach_message_instruction ({leads_with_outreach} leads with outreach messages * 3), total now {expected_credits}")
        
        # show_emails: 3 credits per lead (optional)
        if request.show_emails:
            # Only count leads that actually have emails
            leads_with_emails = 0
            if response.search_results:
                for company_result in response.search_results:
                    if company_result.leads:
                        for lead in company_result.leads:
                            if lead.profile and lead.profile.get_all_emails():
                                leads_with_emails += 1
            email_credits = leads_with_emails * 3
            expected_credits += email_credits
            logger.info(f"Added {email_credits} credits for show_emails ({leads_with_emails} leads with emails * 3), total now {expected_credits}")
        
        # show_phone_numbers: 8 credits per lead (optional)
        if request.show_phone_numbers:
            # Only count leads that actually have phone numbers
            leads_with_phones = 0
            if response.search_results:
                for company_result in response.search_results:
                    if company_result.leads:
                        for lead in company_result.leads:
                            if lead.profile and lead.profile.phone_numbers:
                                leads_with_phones += 1
            phone_credits = leads_with_phones * 8
            expected_credits += phone_credits
            logger.info(f"Added {phone_credits} credits for show_phone_numbers ({leads_with_phones} leads with phones * 8), total now {expected_credits}")
        
        # require_emails: 1 credit per candidate (optional)
        if request.require_emails or request.require_phone_numbers or request.require_phones_or_emails:
            require_email_credits = total_leads * 1
            expected_credits += require_email_credits
            logger.info(f"Added {require_email_credits} credits for require_* ({total_leads} leads * 1), total now {expected_credits}")
        
        # high_freshness: 1 credit per lead (optional)
        if request.high_freshness:
            freshness_credits = total_leads * 1
            expected_credits += freshness_credits
            logger.info(f"Added {freshness_credits} credits for high_freshness ({total_leads} leads * 1), total now {expected_credits}")
    
    logger.info(f"Final expected_credits: {expected_credits}, actual credits_used: {response.credits_used}")
    assert response.credits_used == expected_credits

@pytest.mark.asyncio
async def test_search_company_leads():
    credits1 = await get_credits()
    request = V2SearchCompanyLeadsRequest(
        company_query="ats companies",
        lead_query="c-levels and founders",
        outreach_message_instruction="<300 characters, email style, casual",
        limit=20,
        leads_limit=2,
        show_emails=True,
    )
    generate_curl_command("search_company_leads", request)
    response = await AsyncPearchClient().search_company_leads(request)
    assert any(company_result.score > 0 for company_result in response.search_results)
    assert any(
        lead.profile.linkedin_slug
        for company_result in response.search_results
        for lead in (company_result.leads or [])
    )
    validate_company_leads_credits(request, response)
    credits2 = await get_credits()
    assert credits1 - credits2 == response.credits_used, "Credits check failed"

@pytest.mark.asyncio
async def test_get_search_status():
    start_time = time.time()
    credits1 = await get_credits()
    first_submit_request = V2SearchRequest(
        query="software engineer",
        type="fast",
        limit=2,
    )
    generate_curl_command("search_submit", first_submit_request)
    submit_response = await AsyncPearchClient().search_submit(first_submit_request)
    task_id = submit_response.task_id
    thread_id = submit_response.thread_id
    assert submit_response.status == "pending"    
    status_response = None
    while time.time() - start_time < 180:
        generate_curl_command("get_search_status", {"task_id": task_id})
        status_response: V2SearchStatusResponse = await AsyncPearchClient().get_search_status(task_id)
        assert status_response.task_id == task_id
        assert status_response.status in ["pending", "running", "completed"]
        assert status_response.query == "software engineer"
        if status_response.status == "completed":
            # Get credits2 immediately after completion, before any other operations
            credits2 = await get_credits()
            actual_credits_used = credits1 - credits2
            # Validate that we didn't spend more credits than expected
            assert actual_credits_used == status_response.credits_used, f"Credits check failed: actual={actual_credits_used}, expected={status_response.credits_used}, credits1={credits1}, credits2={credits2}. Actual credits used should not exceed expected."
            # Now validate the credits calculation
            validate_credits(first_submit_request, status_response.result)
            break
        await asyncio.sleep(5)

    second_submit_request = V2SearchRequest(
        thread_id=thread_id,
        limit=4,  # "show more"
    )
    generate_curl_command("search_submit", second_submit_request)
    submit_response = await AsyncPearchClient().search_submit(second_submit_request)
    task_id = submit_response.task_id
    assert submit_response.status == "pending"    
    while time.time() - start_time < 180:
        generate_curl_command("get_search_status", {"task_id": task_id})
        status_response = await AsyncPearchClient().get_search_status(task_id)
        assert status_response.status != "failed"
        if status_response.status == "completed":
            all_results = status_response.result.search_results
            new_results = status_response.result.search_results[2:4]  # Last 2 are new
            status_response.result.search_results = new_results
            validate_credits(second_submit_request, status_response.result, base_request=first_submit_request)
            status_response.result.search_results = all_results
            credits3 = await get_credits()
            assert credits2 - credits3 == status_response.credits_used, "Credits check failed"
            break
        await asyncio.sleep(5)


@pytest.mark.asyncio
async def test_api_call_history():
    request = V1ApiCallHistoryRequest(limit=100)
    generate_curl_command("api_call_history", request)
    response = await AsyncPearchClient().api_call_history(request)
    assert response.api_call_history is not None
    assert len(response.api_call_history) > 0
    assert response.total_credits_used > 0


@pytest.mark.asyncio
async def test_get_user():
    generate_curl_command("get_user", None)
    response: V1UserResponse = await AsyncPearchClient().get_user()
    assert response.user is not None
    assert response.user.email is not None
    assert response.user.api_key is not None
    assert response.credits_remaining is not None
    assert response.pricing is not None
    assert len(response.pricing) > 0
    assert all(pricing.id.endswith("_cost") for pricing in response.pricing)
    assert all(pricing.credits is not None for pricing in response.pricing)
    assert all(pricing.description is not None for pricing in response.pricing)


BASE_URL = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"
API_KEY = os.getenv("PEARCH_API_KEY")
TEST_KEY = os.getenv("PEARCH_TEST_KEY")

 

@pytest.mark.asyncio
async def test_stream_langgraph_api():
    from langgraph_sdk import get_client    
    base_url = BASE_URL.rstrip("/")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if TEST_KEY:
        headers["X-Test-Secret"] = TEST_KEY
    client = get_client(url=base_url, headers=headers)
    thread_id = str(uuid.uuid4())
    assistant_id = "agent"
    input_data = {
        "messages": [
            {
                "type": "human",
                "content": "ml engineers in seattle"
            }
        ],
        "selected_profiles_list": None,
        "selected_company_slugs": None,
        "selected_full_profile_slug": None
    }
    config = {
        "recursion_limit": 100
    }
    chunks_received = []
    stream_mode = ["updates", "values", "custom"]
    generate_threads_runs_stream_curl(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input_data=input_data,
        config=config,
        stream_mode=stream_mode,
        stream_resumable=True,
        on_disconnect="continue"
    )
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input=input_data,
        config=config,
        stream_mode=stream_mode,
        stream_resumable=True,
        on_disconnect="continue"
    ):
        chunks_received.append(chunk)
        logger.info(f"Chunk: {chunk}")
    assert len(chunks_received) > 0
    

 
