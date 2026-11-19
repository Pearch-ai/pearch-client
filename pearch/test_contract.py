"""What a live Pearch API must deliver to a client.

Every test calls the API through AsyncPearchClient and fails on any error response.
Run the cheapest check with `pytest -m contract_health pearch/test_contract.py`
and the whole public surface with `-m contract_full`. They need PEARCH_API_KEY and spend credits.
"""

import asyncio
import time
import uuid

import pytest

from pearch.client import AsyncPearchClient
from pearch.schema import (
    CustomFilters,
    Job,
    V1ApiCallHistoryRequest,
    V1DeleteJobsRequest,
    V1FindMatchingJobsRequest,
    V1ListJobsRequest,
    V1ProfileBatchItem,
    V1ProfileBatchRequest,
    V1ProfileRequest,
    V1SearchRequest,
    V1UpsertJobsRequest,
    V2CompanyRequest,
    V2SearchCompanyLeadsRequest,
    V2SearchCountRequest,
    V2SearchRequest,
)

pytestmark = pytest.mark.asyncio

QUERY = "Software engineer with Python experience from Florida who is interested in science"
PRO_SEARCH_DEADLINE_SECONDS = 900


@pytest.fixture
def client(request) -> AsyncPearchClient:
    options = request.config.hook.pytest_pearch_client_options() or {}
    return AsyncPearchClient(**{"timeout": 180, "max_retries": 0, **options})


def assert_search_results(response) -> None:
    assert response.status == "Done"
    assert response.uuid
    assert response.thread_id
    assert response.search_results
    assert all(result.docid and result.profile for result in response.search_results)


@pytest.mark.contract_health
@pytest.mark.contract_full
async def test_fast_search_returns_profiles(client):
    request = V2SearchRequest(
        query=QUERY,
        type="fast",
        limit=1,
        insights=False,
        high_freshness=False,
        profile_scoring=False,
        reveal_emails=False,
        reveal_phones=False,
        strict_filters=False,
    )
    async with client:
        response = await client.search(request)
    assert_search_results(response)


@pytest.mark.contract_full
async def test_submitted_pro_search_returns_scored_profiles_with_insights(client):
    request = V2SearchRequest(
        query=QUERY,
        type="pro",
        time_budget=60,
        limit=3,
        insights=True,
        high_freshness=False,
        profile_scoring=True,
        reveal_emails=False,
        reveal_phones=False,
        strict_filters=False,
    )
    async with client:
        submitted = await client.search_submit(request)
        assert submitted.task_id
        assert submitted.status == "pending"
        deadline = time.monotonic() + PRO_SEARCH_DEADLINE_SECONDS
        status = await client.get_search_status(submitted.task_id)
        while status.status not in ("completed", "failed") and time.monotonic() < deadline:
            await asyncio.sleep(5)
            status = await client.get_search_status(submitted.task_id)
    assert status.status == "completed", status.error or status.status
    assert_search_results(status.result)
    assert any(result.score is not None for result in status.result.search_results)
    assert any(result.insights is not None for result in status.result.search_results)


@pytest.mark.contract_full
async def test_search_count_finds_matching_profiles(client):
    request = V2SearchCountRequest(
        custom_filters=CustomFilters(locations=["United States"], titles=["software engineer"]),
        stats=True,
    )
    async with client:
        response = await client.search_count(request)
    assert response.count > 0
    assert response.uuid
    assert response.stats is not None


@pytest.mark.contract_full
async def test_company_leads_search_returns_companies_with_leads(client):
    request = V2SearchCompanyLeadsRequest(
        company_query="ats companies",
        lead_query="c-levels and founders",
        limit=5,
        leads_limit=1,
        reveal_emails=False,
        reveal_phones=False,
        high_freshness=False,
        company_high_freshness=False,
    )
    async with client:
        response = await client.search_company_leads(request)
    assert response.search_results
    assert any(result.company for result in response.search_results)
    assert any(result.leads for result in response.search_results)


@pytest.mark.contract_full
async def test_company_is_found_by_linkedin_slug_and_by_domain(client):
    async with client:
        by_slug = await client.get_company(V2CompanyRequest(linkedin_slug="openai"))
        by_domain = await client.get_company(V2CompanyRequest(domain="openai.com"))
    assert by_slug.company.linkedin_slug == "openai"
    assert by_slug.credits_used is not None
    assert by_domain.company.domain == "openai.com" or by_domain.company.linkedin_slug == "openai"
    assert by_domain.credits_used is not None


@pytest.mark.contract_full
async def test_v1_search_returns_profiles(client):
    async with client:
        profiles = await client.search_v1(V1SearchRequest(query="software engineer", type="fast", limit=3))
    assert profiles
    assert all(profile.linkedin_slug for profile in profiles)


@pytest.mark.contract_full
async def test_job_is_upserted_listed_matched_and_deleted(client):
    job_id = f"contract-test-{uuid.uuid4().hex}"
    async with client:
        upserted = await client.upsert_jobs(
            V1UpsertJobsRequest(
                jobs=[
                    Job(
                        job_id=job_id,
                        job_description="Senior software engineer working with Python and distributed systems",
                    )
                ]
            )
        )
        try:
            assert upserted.status and "success" in upserted.status.lower()

            listed = await client.list_jobs(V1ListJobsRequest(limit=100))
            assert listed.jobs is not None
            assert any(job.job_id == job_id for job in listed.jobs)

            matched = None
            for _ in range(3):
                matched = await client.find_matching_jobs(
                    V1FindMatchingJobsRequest(
                        profile={
                            "description": "Senior software engineer experienced in Python and distributed systems"
                        },
                        limit=2,
                    )
                )
                if matched.jobs and any(job.job_id == job_id for job in matched.jobs):
                    break
                await asyncio.sleep(2)
            assert matched.jobs
            assert any(job.job_id == job_id for job in matched.jobs)
        finally:
            deleted = await client.delete_jobs(V1DeleteJobsRequest(job_ids=[job_id]))
            assert deleted.deleted_count == 1


@pytest.mark.contract_full
async def test_profile_with_high_freshness_reveals_contacts(client):
    request = V1ProfileRequest(
        docid="victorsunden",
        high_freshness=True,
        reveal_emails=True,
        reveal_phones=True,
        with_profile=True,
    )
    async with client:
        response = await client.get_profile(request)
    assert response.profile.linkedin_slug
    assert response.credits_used is not None
    assert response.profile.personal_emails or response.profile.business_emails
    assert response.profile.phone_numbers


@pytest.mark.contract_full
async def test_batch_enrichment_returns_every_profile(client):
    request = V1ProfileBatchRequest(
        profiles=[
            V1ProfileBatchItem(reference_id="vlad", docid="vslaykovsky"),
            V1ProfileBatchItem(reference_id="victor", docid="victorsunden"),
        ],
        with_profile=True,
    )
    async with client:
        response = await client.enrich_profiles(request)
    assert response.profiles_succeeded == 2
    assert response.profiles_failed == 0
    assert all(result.http == 200 for result in response.results)


@pytest.mark.contract_full
async def test_api_call_history_lists_calls(client):
    async with client:
        response = await client.api_call_history(V1ApiCallHistoryRequest(limit=20))
    assert response.api_call_history
    assert response.total_credits_used is not None


@pytest.mark.contract_full
async def test_user_reports_credits_and_pricing(client):
    async with client:
        response = await client.get_user()
    assert response.user is not None
    assert response.credits_remaining is not None
    assert response.pricing
