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
        "search_v1": ("GET", "v1/search"),
        "upsert_jobs": ("POST", "v1/upsert_jobs"),
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
    response: V1FindMatchingJobsResponse = await AsyncPearchClient().find_matching_jobs(request)
    assert response.jobs
    assert any(job.job_id for job in response.jobs)
    assert response.credits_used == len(response.jobs) * 1


@pytest.mark.asyncio
async def test_profile():
    request = V1ProfileRequest(docid="vslaykovsky", show_emails=True, high_freshness=True, show_phone_numbers=True)
    generate_curl_command("get_profile", request)
    response: V1ProfileResponse = await AsyncPearchClient().get_profile(request)
    assert "vlad" in response.profile.first_name.lower()
    assert response.credits_used == 2 + 2 * (0 if not response.profile.get_all_emails() else 1) + 14 * (0 if not response.profile.all_phone_numbers() else 1)

    request = V1ProfileRequest(docid="victorsunden", show_emails=True, high_freshness=True, show_phone_numbers=True)
    generate_curl_command("get_profile", request)
    response: V1ProfileResponse = await AsyncPearchClient().get_profile(request)
    assert "victor" in response.profile.first_name.lower()
    assert response.credits_used == 2 + 2 * (0 if not response.profile.get_all_emails() else 1) + 14 * (0 if not response.profile.all_phone_numbers() else 1)

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
    response: V1UpsertJobsResponse = await AsyncPearchClient().upsert_jobs(request)
    assert response.credits_used == 1
    assert "success" in response.status.lower()


def validate_credits(request: V2SearchRequest, response: V2SearchResponse | V2SearchStatusResponse):
    expected_credits = 0   
    for result in response.search_results:
        candidate_credits = 0
        profile_id = result.profile.linkedin_slug if result.profile and result.profile.linkedin_slug else "unknown"
        if request.type == "pro":
            candidate_credits += 5
            logger.info(f"{profile_id}: Incremented candidate_credits by 5 for type 'pro', total now {candidate_credits}")
        elif request.type == "fast":
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for type 'fast', total now {candidate_credits}")
        if request.insights and result.insights:
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for insights, total now {candidate_credits}")
        if request.high_freshness:
            candidate_credits += 2
            logger.info(f"{profile_id}: Incremented candidate_credits by 2 for high_freshness, total now {candidate_credits}")
        if request.profile_scoring and result.score is not None:
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for profile_scoring, total now {candidate_credits}")
        if request.show_emails and result.profile and result.profile.get_all_emails():
            candidate_credits += 2
            logger.info(f"{profile_id}: Incremented candidate_credits by 2 for show_emails, total now {candidate_credits}")
        if request.show_phone_numbers and result.profile and result.profile.phone_numbers:
            candidate_credits += 14
            logger.info(f"{profile_id}: Incremented candidate_credits by 14 for show_phone_numbers, total now {candidate_credits}")
        if request.require_emails or request.require_phone_numbers:
            candidate_credits += 1
            logger.info(f"{profile_id}: Incremented candidate_credits by 1 for require_emails or require_phone_numbers, total now {candidate_credits}")
        expected_credits += candidate_credits
        logger.info(f"{profile_id}: Added candidate_credits {candidate_credits} to expected_credits, expected_credits now {expected_credits}")
    assert response.credits_used == expected_credits    


@pytest.mark.asyncio
async def test_v2_pro_search_generic():
    request = V2SearchRequest(
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
    generate_curl_command("search", request)
    response: V2SearchResponse = await AsyncPearchClient().search(request)
    assert any(result.profile.linkedin_slug for result in response.search_results)
    validate_credits(request, response)

@pytest.mark.asyncio
async def test_v2_pro_search_narrow():
    request = V2SearchRequest(query="employees of collectly", limit=2)
    generate_curl_command("search", request)
    response = await AsyncPearchClient().search(request)
    assert response.search_results[0].profile.linkedin_slug is not None
    validate_credits(request, response)


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
        if request.require_emails or request.require_phone_numbers:
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
    validate_company_leads_credits(request, response)


@pytest.mark.asyncio
async def test_get_search_status():
    start_time = time.time()
    submit_request = V2SearchRequest(
        query="test query for status check",
        type="fast",
        limit=2,
    )
    submit_response = await AsyncPearchClient().search_submit(submit_request)
    task_id = submit_response.task_id
    assert submit_response.status == "pending"    
    while time.time() - start_time < 180:
        generate_curl_command("get_search_status", {"task_id": task_id})
        status_response = await AsyncPearchClient().get_search_status(task_id)
        assert status_response.task_id == task_id
        assert status_response.status in ["pending", "running", "completed"]
        assert status_response.query == "test query for status check"
        if status_response.status == "completed":
            validate_credits(submit_request, status_response.result)
            break
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_api_call_history():
    request = V1ApiCallHistoryRequest(limit=5)
    generate_curl_command("api_call_history", request)
    response = await AsyncPearchClient().api_call_history(request)
    assert response.api_call_history is not None
    assert isinstance(response.api_call_history, list) 