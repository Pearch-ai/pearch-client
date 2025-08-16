#!/usr/bin/env python3
"""Parallel API endpoint testing script.
Tests multiple endpoints concurrently with configurable host and API key.
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict

import dotenv

from pearch.client import AsyncPearchClient
from pearch.schema import (
    V1FindMatchingJobsRequest,
    V1ProfileRequest,
    V1SearchRequest,
    V1UpsertJobsRequest,
    V2SearchRequest,
    V2SearchCompanyLeadsRequest,
    Job
)

dotenv.load_dotenv()

log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('api_test')



async def test_find_matching_jobs(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "find_matching_jobs"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V1FindMatchingJobsRequest(
        profile={
            "description": "I'm a Senior Silicon Design Engineer with strong background in computer engineering, hands-on silicon design experience, and deep understanding of the semiconductor industry."
        },
        limit=2
    )
    
    error_msg = None
    
    
    try:
        response = await client.find_matching_jobs(request)
        
        assert any(job.job_id for job in response.jobs)
        
        logger.info(f"Test {test_name} - SUCCESS")
        
    except Exception as e:
        
        error_msg = str(e)
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    logger.info(f"Test {test_name} completed (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "find_matching_jobs",
        "client_method": "find_matching_jobs",
        "request": request.model_dump(),        
        "duration": round(duration, 3),
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("find_matching_jobs", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result


async def test_profile(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "profile"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V1ProfileRequest(
        docid="vslaykovsky",
        show_emails=True
    )
    error_msg = None
    try:
        response = await client.get_profile(request)
        assert "vlad" in response.profile.first_name.lower()        
        logger.info(f"Test {test_name} - SUCCESS")
        
    except Exception as e:
        
        error_msg = str(e)
        
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    logger.info(f"Test {test_name} completed:  (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "get_profile",
        "client_method": "get_profile",
        "request": request.model_dump(),
        
        
        
        "duration": round(duration, 3),
        
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("get_profile", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result


async def test_v1_fast_search(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "v1_fast_search"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V1SearchRequest(
        query="software engineer",
        limit=100,
        type="fast"
    )
    
    
    error_msg = None
    
    
    try:
        response = await client.search_v1(request)
        
        assert any(result.linkedin_slug for result in response)
        
        logger.info(f"Test {test_name} - SUCCESS")
        
    except Exception as e:
        
        error_msg = str(e)
        
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    logger.info(f"Test {test_name} completed:  (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "search_v1",
        "client_method": "search_v1",
        "request": request.model_dump(),
        
        
        
        "duration": round(duration, 3),
        
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("search_v1", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result





async def test_upsert_jobs(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "upsert_jobs"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V1UpsertJobsRequest(
        jobs=[Job(job_id="1", job_description="software engineer in test for la ai startup")]
    )
    
    
    error_msg = None
    
    
    
    try:
        response = await client.upsert_jobs(request)        
        assert "success" in response.status.lower()
        logger.info(f"Test {test_name} - SUCCESS")        
    except Exception as e:
        
        error_msg = str(e)
        
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    
    logger.info(f"Test {test_name} completed: (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "upsert_jobs",
        "client_method": "upsert_jobs",
        "request": request.model_dump(),
        
        
        
        "duration": round(duration, 3),
        
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("upsert_jobs", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result


async def test_v2_pro_search(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "v2_pro_search"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V2SearchRequest(
        query="Find me engineers in California speaking at least basic english working in software industry with experience at FAANG with 2+ years of experience and at least 500 followers and at least BS degree",
        limit=2
    )
        
    error_msg = None
    
    
    try:
        response = await client.search(request)

        assert any(result.profile.linkedin_slug for result in response.search_results)
        
        
        logger.info(f"Test {test_name} - SUCCESS")
        
    except Exception as e:
        
        error_msg = str(e)
        
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    logger.info(f"Test {test_name} completed:  (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "search",
        "client_method": "search",
        "request": request.model_dump(),        
        "duration": round(duration, 3),        
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("search", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result


async def test_v2_pro_search_narrow(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "v2_pro_search_narrow"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V2SearchRequest(
        query="employees of collectly",
        limit=2
    )
        
    error_msg = None
    
    
    try:
        response = await client.search(request)        
        assert response.search_results[0].profile.linkedin_slug is not None        
        logger.info(f"Test {test_name} - SUCCESS")
    except Exception as e:
        
        error_msg = str(e)
        
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    logger.info(f"Test {test_name} completed:  (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "search",
        "client_method": "search",
        "request": request.model_dump(),
         
        
        "duration": round(duration, 3),        
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("search", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result


async def test_search_company_leads(client: AsyncPearchClient, show_curl: bool = False) -> Dict[str, Any]:
    test_name = "search_company_leads"
    logger.info(f"Starting test: {test_name}")
    start_time = time.time()
    
    request = V2SearchCompanyLeadsRequest(
        company_query="ats companies",
        lead_query="c-levels and founders",
        outreach_message_instruction="<300 characters, email style, casual",
        limit=12,
        leads_limit=2,
        show_emails=True
    )
    
    
    error_msg = None
    
    
    try:
        response = await client.search_company_leads(request)
        
        assert any(
            any(lead.profile.linkedin_slug for lead in company_result.leads or [])
            for company_result in response.search_results
        )        
        
        logger.info(f"Test {test_name} - SUCCESS")
        
    except Exception as e:
             
        error_msg = str(e)        
        logger.error(f"Test {test_name} - Error: {error_msg}")
    
    duration = time.time() - start_time
    
    logger.info(f"Test {test_name} completed:  (duration: {duration:.3f}s)")
    
    result = {
        "test_name": test_name,
        "method": "search_company_leads",
        "client_method": "search_company_leads",
        "request": request.model_dump(),
        "duration": round(duration, 3),
        "error": error_msg
    }
    
    if show_curl:
        curl_command = generate_curl_command("search_company_leads", request, client.base_url, client.api_key)
        result["curl_command"] = curl_command
    
    return result


def get_available_tests():
    return {
        "find_matching_jobs": test_find_matching_jobs,
        "profile": test_profile,
        "v1_fast_search": test_v1_fast_search,
        "upsert_jobs": test_upsert_jobs,
        "v2_pro_search": test_v2_pro_search,
        "v2_pro_search_narrow": test_v2_pro_search_narrow,
        "search_company_leads": test_search_company_leads,
    }


def generate_curl_command(client_method: str, request: Any, base_url: str, api_key: str) -> str:
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
    url = f"{base_url.rstrip('/')}/{endpoint}"
    
    curl_parts = ["curl", "-X", http_method]
    curl_parts.extend(["-H", f"'Authorization: Bearer {api_key}'"])
    curl_parts.extend(["-H", "'Content-Type: application/json'"])
    
    if http_method == "POST" and hasattr(request, 'model_dump'):
        request_json = json.dumps(request.model_dump(exclude_none=True))
        curl_parts.extend(["-d", f"'{request_json}'"])
    elif http_method == "GET" and hasattr(request, 'model_dump'):
        params = request.model_dump(exclude_none=True)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            url += f"?{query_string}"
    
    curl_parts.append(f"'{url}'")
    
    return " ".join(curl_parts)




async def main():
    parser = argparse.ArgumentParser(description="Test API endpoints in parallel")
    parser.add_argument("--host", default="https://api.pearch.ai", 
                       help="Host URL. Default: https://api.pearch.ai")
    parser.add_argument("--api_key", 
                       help="API key for authorization. Find it at https://pearch.ai")
    parser.add_argument("--test", 
                       help="Run a specific test by name. Use --list-tests to see available tests.")
    parser.add_argument("--list-tests", action="store_true",
                       help="List all available test names and exit")
    parser.add_argument("--curl", action="store_true",
                       help="Show curl commands for reproducing requests")
    
    args = parser.parse_args()
    
    available_tests = get_available_tests()
    
    # Handle --list-tests option
    if args.list_tests:
        logger.info("Available tests:")
        for test_name in available_tests.keys():
            logger.info(f"  {test_name}")
        sys.exit(0)
    
    # Filter test functions if --test is specified
    tests_to_run = available_tests
    if args.test:
        if args.test not in available_tests:
            logger.error(f"Error: Test '{args.test}' not found.")
            logger.info("Available tests:")
            for test_name in available_tests.keys():
                logger.info(f"  {test_name}")
            sys.exit(1)
        tests_to_run = {args.test: available_tests[args.test]}
    
    # Resolve host aliases
    host = args.host
    
    logger.info("Starting API endpoint tests")
    logger.info(f"Target host: {host}")
    logger.info(f"API key: {args.api_key[:10]}...")
    if args.test:
        logger.info(f"Running single test: {args.test}")
    logger.info(f"Using API key: {args.api_key[:10]}...")
    logger.info("-" * 60)
    
    start_time = time.time()
    
    logger.info(f"Starting {len(tests_to_run)} tests in parallel")
    
    # Create client and run all tests in parallel
    client = AsyncPearchClient(
        api_key=args.api_key,
        base_url=host
    )
    
    try:
        tasks = [
            test_func(client, args.curl)
            for test_func in tests_to_run.values()
        ]
        
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()
    
    total_time = time.time() - start_time
    logger.info(f"All tests completed in {total_time:.3f}s")
    
    # Print results
    logger.info("Test Results:")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for result in results:
        # Check if status is in 2xx range (200-299)
        status_text = "PASS" if not result["error"]  else "FAIL"
        
        logger.info(f" {result['test_name']} - {status_text}")
        logger.info(f"   Client Method: {result['client_method']}")
        logger.info(f"   Request: {str(result['request'])[:100]}{'...' if len(str(result['request'])) > 100 else ''}")
        logger.info(f"   Duration: {result['duration']}s")
        if result.get("error"):
            logger.error(f"   Error: {result['error']}")
        
        # Print curl command only if --curl flag is provided
        if args.curl and 'curl_command' in result:
            logger.info("   Reproduce with curl:")
            logger.info(f"   {result['curl_command']}")
        
        # Check if status is in 2xx range (200-299)
        if not result["error"]:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"Summary: {passed} passed, {failed} failed")
    logger.info(f"Total execution time: {round(total_time, 3)}s")
    
    
    if failed > 0:
        logger.error(f"Tests failed: {failed} out of {passed + failed}")
    else:
        logger.info("All tests passed successfully!")
    
    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
