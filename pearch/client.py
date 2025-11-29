import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from .schema import (
    Profile,
    V1FindMatchingJobsRequest,
    V1FindMatchingJobsResponse,
    V1ProfileRequest,
    V1ProfileResponse,
    V1ApiCallHistoryRequest,
    V1ApiCallHistoryResponse,
    V1SearchRequest,
    V1ProSearchResponse,
    V1UpsertJobsRequest,
    V1UpsertJobsResponse,
    V1ListJobsRequest,
    V1ListJobsResponse,
    V1DeleteJobsRequest,
    V1DeleteJobsResponse,
    V1UserResponse,
    V2SearchCompanyLeadsRequest,
    V2SearchCompanyLeadsResponse,
    V2SearchRequest,
    V2SearchResponse,
    V2SearchSubmitResponse,
    V2SearchStatusResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)

TIMEOUT = 1800


class PearchAPIError(Exception):
    """Base exception for Pearch API errors"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class PearchAuthenticationError(PearchAPIError):
    """Raised when authentication fails"""

    pass


class PearchValidationError(PearchAPIError):
    """Raised when request validation fails"""

    pass


class PearchClient:
    """
    Synchronous client for the Pearch.AI API

    This client provides access to all Pearch.AI API endpoints with proper
    type hints, error handling, and authentication management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = None,
        timeout: float = TIMEOUT,
        max_retries: int = 3,
        token: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Pearch API client

        Args:
            api_key: Your Pearch.AI API key (defaults to PEARCH_API_KEY environment variable)
            base_url: Base URL for the API (default: PEARCH_API_URL or https://api.pearch.ai/)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
            token: Optional test token (sent as X-Test-Secret header)
            **kwargs: Additional arguments passed to httpx.Client
        """
        # Use environment variable if api_key not provided
        if api_key is None:
            api_key = os.getenv("PEARCH_API_KEY")

        if base_url is None:
            base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"

        if not api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set the PEARCH_API_KEY environment variable."
            )

        self.api_key = api_key
        self.token = token
        self.base_url = base_url.rstrip("/") + "/"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "pearch-python-client/1.0",
        }

        if token:
            headers["X-Test-Secret"] = token

        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers=headers, **kwargs
        )

        # Configure retries
        transport = httpx.HTTPTransport(retries=max_retries)
        self._client._transport = transport

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data for POST requests
            params: Query parameters for GET requests

        Returns:
            Parsed JSON response

        Raises:
            PearchAuthenticationError: When API key is invalid
            PearchValidationError: When request parameters are invalid
            PearchAPIError: For other API errors
        """
        logger.debug(f"Making {method} request to {endpoint}")

        response = self._client.request(
            method=method, url=endpoint, json=data, params=params
        )

        # Handle different HTTP status codes
        if response.status_code == 401:
            raise PearchAuthenticationError(
                "Invalid API key",
                status_code=401,
                response_data=response.json() if response.content else {},
            )
        elif response.status_code == 400:
            error_data = response.json() if response.content else {}
            raise PearchValidationError(
                f"Invalid request parameters: {error_data}",
                status_code=400,
                response_data=error_data,
            )
        elif response.status_code >= 400:
            error_data = response.json() if response.content else {}
            raise PearchAPIError(
                f"API request failed with status {response.status_code}: {error_data}",
                status_code=response.status_code,
                response_data=error_data,
            )

        response.raise_for_status()
        return response.json()

    # V2 API Methods

    def search(self, request: V2SearchRequest) -> V2SearchResponse:
        """
        Execute an enhanced search query (v2 endpoint)

        This is the recommended search endpoint with advanced parameters for
        granular control over search behavior.

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing v2 search with query: {request.query}")

        response_data = self._make_request(
            method="POST",
            endpoint="v2/search",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchResponse(**response_data)

    def search_company_leads(
        self, request: V2SearchCompanyLeadsRequest
    ) -> V2SearchCompanyLeadsResponse:
        """
        Search for companies and their leads

        Find companies based on a query and optionally find leads within those
        companies with personalized outreach messages.

        Args:
            request: Company and lead search parameters

        Returns:
            Company search response with leads
        """
        logger.info(f"Searching company leads with query: {request.company_query}")

        response_data = self._make_request(
            method="POST",
            endpoint="v2/search_company_leads",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchCompanyLeadsResponse(**response_data)

    def search_submit(self, request: V2SearchRequest) -> V2SearchSubmitResponse:
        """
        Submit a search task for background execution

        Submit a search query to be processed asynchronously. Returns a task ID
        and thread ID that can be used to check the status, retrieve results,
        and paginate through search results.

        Args:
            request: Search submit request parameters

        Returns:
            Search submit response with task ID, thread ID, and status
        """
        logger.info(f"Submitting search task with query: {request.query}")

        response_data = self._make_request(
            method="POST",
            endpoint="v2/search/submit",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchSubmitResponse(**response_data)

    def get_search_status(self, task_id: str) -> V2SearchStatusResponse:
        """
        Get the status of a submitted search task

        Check the current status of a search task and retrieve results if completed.

        Args:
            task_id: The task ID returned from search_submit

        Returns:
            Search status response with current status and results if available
        """
        logger.info(f"Checking status for search task: {task_id}")

        response_data = self._make_request(
            method="GET",
            endpoint=f"v2/search/status/{task_id}",
        )

        return V2SearchStatusResponse(**response_data)

    # V1 API Methods (Legacy)

    def search_v1(self, request: V1SearchRequest) -> V1ProSearchResponse:
        """
        Execute a search query (v1 legacy endpoint)

        Legacy search endpoint that accepts parameters as query parameters.
        Consider using the v2 search endpoint for new implementations.

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing v1 search with query: {request.query}")

        response_data = self._make_request(
            method="GET",
            endpoint="v1/search",
            params=request.model_dump(exclude_none=True),
        )

        return V1ProSearchResponse(**response_data)

    def upsert_jobs(self, request: V1UpsertJobsRequest) -> V1UpsertJobsResponse:
        """
        Upload or update job listings

        Submit a list of jobs to be indexed. If job_id already exists, it will
        be updated. Supports both upsert and replace modes.

        Args:
            request: Job upsert request with job listings

        Returns:
            Upsert response with processing status
        """
        logger.info(f"Upserting {len(request.jobs)} jobs (replace={request.replace})")

        response_data = self._make_request(
            method="POST",
            endpoint="v1/upsert_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1UpsertJobsResponse(**response_data)

    def list_jobs(self, request: V1ListJobsRequest) -> V1ListJobsResponse:
        """
        List all jobs in the user's job index
        
        Args:
            request: List jobs request with optional limit
            
        Returns:
            List jobs response with job data
        """
        logger.info(f"Listing jobs with limit: {request.limit}")
        
        response_data = self._make_request(
            method="GET",
            endpoint="v1/list_jobs",
            params=request.model_dump(exclude_none=True),
        )
        
        return V1ListJobsResponse(**response_data)
    
    def delete_jobs(self, request: V1DeleteJobsRequest) -> V1DeleteJobsResponse:
        """
        Delete jobs from the user's job index
        
        Args:
            request: Delete jobs request with job IDs to delete
            
        Returns:
            Delete jobs response with deletion status
        """
        logger.info(f"Deleting {len(request.job_ids)} jobs")
        
        response_data = self._make_request(
            method="POST",
            endpoint="v1/delete_jobs",
            data=request.model_dump(exclude_none=True),
        )
        
        return V1DeleteJobsResponse(**response_data)

    def find_matching_jobs(
        self, request: V1FindMatchingJobsRequest
    ) -> V1FindMatchingJobsResponse:
        """
        Find jobs relevant to a candidate profile

        Accepts a profile in arbitrary JSON format and returns a list of
        relevant jobs with match scores and insights.

        Args:
            request: Job matching request with candidate profile

        Returns:
            Matching jobs response with relevance scores
        """
        logger.info(f"Finding matching jobs for profile (limit={request.limit})")

        response_data = self._make_request(
            method="POST",
            endpoint="v1/find_matching_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1FindMatchingJobsResponse(**response_data)

    def get_profile(self, request: V1ProfileRequest) -> V1ProfileResponse:
        """
        Retrieve a user profile by LinkedIn profile ID

        Get detailed profile information for a specific LinkedIn profile ID
        with optional real-time updates and contact information.

        Args:
            request: Profile request with docid and options

        Returns:
            Profile response with detailed candidate information
        """
        logger.info(f"Retrieving profile for docid: {request.docid}")

        response_data = self._make_request(
            method="GET",
            endpoint="v1/profile",
            params=request.model_dump(exclude_none=True),
        )

        return V1ProfileResponse(**response_data)

    def api_call_history(self, request: V1ApiCallHistoryRequest) -> V1ApiCallHistoryResponse:
        """
        Retrieve API call history for the authenticated user

        Get a list of previous API calls including their parameters,
        results, and metadata.

        Args:
            request: API call history request with limit and optional paths parameters

        Returns:
            API call history response with list of previous API calls and total credits used
        """
        logger.info(f"Retrieving API call history with limit: {request.limit}")

        params = request.model_dump(exclude_none=True)
        if params.get("paths"):
            params["paths"] = ",".join(params["paths"])
        response_data = self._make_request(
            method="GET",
            endpoint="v1/api_call_history",
            params=params,
        )

        return V1ApiCallHistoryResponse(**response_data)

    def get_user(self) -> V1UserResponse:
        """
        Get user information, remaining credits, and pricing details

        Retrieves information about the authenticated user including their
        email, API key details, remaining credits, and current pricing
        configuration for different operations.

        Returns:
            User response with user info, credits remaining, and pricing details
        """
        logger.info("Retrieving user information")

        response_data = self._make_request(
            method="GET",
            endpoint="v1/user"
        )

        return V1UserResponse(**response_data)


class AsyncPearchClient:
    """
    Asynchronous client for the Pearch.AI API

    This client provides async access to all Pearch.AI API endpoints with proper
    type hints, error handling, and authentication management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = TIMEOUT,
        max_retries: int = 3,
        token: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the async Pearch API client

        Args:
            api_key: Your Pearch.AI API key (defaults to PEARCH_API_KEY environment variable)
            base_url: Base URL for the API (default: PEARCH_API_URL or https://api.pearch.ai/)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
            token: Optional test token (sent as X-Test-Secret header)
            **kwargs: Additional arguments passed to httpx.AsyncClient
        """
        # Use environment variable if api_key not provided
        if base_url is None:
            base_url = os.getenv("PEARCH_API_URL") or "https://api.pearch.ai/"

        if api_key is None:
            api_key = os.getenv("PEARCH_API_KEY")

        if not api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set the PEARCH_API_KEY environment variable."
            )
        if token is None:
            token = os.getenv("PEARCH_TEST_KEY")

        self.api_key = api_key
        self.token = token
        self.base_url = base_url.rstrip("/") + "/"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "pearch-python-client/1.0",
        }

        if token:
            headers["X-Test-Secret"] = token

        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers=headers, **kwargs
        )

        # Configure retries
        transport = httpx.AsyncHTTPTransport(retries=max_retries)
        self._client._transport = transport

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data for POST requests
            params: Query parameters for GET requests

        Returns:
            Parsed JSON response

        Raises:
            PearchAuthenticationError: When API key is invalid
            PearchValidationError: When request parameters are invalid
            PearchAPIError: For other API errors
        """
        logger.debug(f"Making async {method} request to {endpoint}")

        response = await self._client.request(
            method=method, url=endpoint, json=data, params=params
        )

        # Handle different HTTP status codes
        if response.status_code == 401:
            raise PearchAuthenticationError(
                "Invalid API key",
                status_code=401,
                response_data=response.content,
            )
        elif response.status_code == 400:
            error_data = response.content
            raise PearchValidationError(
                f"Invalid request parameters: {error_data}",
                status_code=400,
                response_data=error_data,
            )
        elif response.status_code >= 400:
            error_data = response.content
            raise PearchAPIError(
                f"API request failed with status {response.status_code}: {error_data}",
                status_code=response.status_code,
                response_data=error_data,
            )

        response.raise_for_status()
        return response.json()

    # V2 API Methods

    async def search(self, request: V2SearchRequest) -> V2SearchResponse:
        """
        Execute an enhanced search query (v2 endpoint) - async version

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing async v2 search with query: {request.query}")

        response_data = await self._make_request(
            method="POST",
            endpoint="v2/search",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchResponse(**response_data)

    async def search_company_leads(
        self, request: V2SearchCompanyLeadsRequest
    ) -> V2SearchCompanyLeadsResponse:
        """
        Search for companies and their leads - async version

        Args:
            request: Company and lead search parameters

        Returns:
            Company search response with leads
        """
        logger.info(
            f"Searching company leads async with query: {request.company_query}"
        )

        response_data = await self._make_request(
            method="POST",
            endpoint="v2/search_company_leads",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchCompanyLeadsResponse(**response_data)

    async def search_submit(self, request: V2SearchRequest) -> V2SearchSubmitResponse:
        """
        Submit a search task for background execution - async version

        Submit a search query to be processed asynchronously. Returns a task ID
        and thread ID that can be used to check the status, retrieve results,
        and paginate through search results.

        Args:
            request: Search submit request parameters

        Returns:
            Search submit response with task ID, thread ID, and status
        """
        logger.info(f"Submitting async search task with query: {request.query}")

        response_data = await self._make_request(
            method="POST",
            endpoint="v2/search/submit",
            data=request.model_dump(exclude_none=True),
        )

        return V2SearchSubmitResponse(**response_data)

    async def get_search_status(self, task_id: str) -> V2SearchStatusResponse:
        """
        Get the status of a submitted search task - async version

        Check the current status of a search task and retrieve results if completed.

        Args:
            task_id: The task ID returned from search_submit

        Returns:
            Search status response with current status and results if available
        """
        logger.info(f"Checking async status for search task: {task_id}")

        response_data = await self._make_request(
            method="GET",
            endpoint=f"v2/search/status/{task_id}",
        )

        return V2SearchStatusResponse(**response_data)

    # V1 API Methods (Legacy)

    async def search_v1(self, request: V1SearchRequest) -> List[Profile]:
        """
        Execute a search query (v1 legacy endpoint) - async version

        Args:
            request: Search request parameters

        Returns:
            Search response with results and metadata
        """
        logger.info(f"Executing async v1 search with query: {request.query}")

        response_data = await self._make_request(
            method="GET",
            endpoint="v1/search",
            params=request.model_dump(exclude_none=True),
        )
        return [Profile(**result) for result in response_data]

    async def upsert_jobs(self, request: V1UpsertJobsRequest) -> V1UpsertJobsResponse:
        """
        Upload or update job listings - async version

        Args:
            request: Job upsert request with job listings

        Returns:
            Upsert response with processing status
        """
        logger.info(
            f"Upserting {len(request.jobs)} jobs async (replace={request.replace})"
        )

        response_data = await self._make_request(
            method="POST",
            endpoint="v1/upsert_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1UpsertJobsResponse(**response_data)

    async def list_jobs(self, request: V1ListJobsRequest) -> V1ListJobsResponse:
        """
        List all jobs in the user's job index - async version
        
        Args:
            request: List jobs request with optional limit
            
        Returns:
            List jobs response with job data
        """
        logger.info(f"Listing jobs async with limit: {request.limit}")
        
        response_data = await self._make_request(
            method="GET",
            endpoint="v1/list_jobs",
            params=request.model_dump(exclude_none=True),
        )
        
        return V1ListJobsResponse(**response_data)
    
    async def delete_jobs(self, request: V1DeleteJobsRequest) -> V1DeleteJobsResponse:
        """
        Delete jobs from the user's job index - async version
        
        Args:
            request: Delete jobs request with job IDs to delete
            
        Returns:
            Delete jobs response with deletion status
        """
        logger.info(f"Deleting {len(request.job_ids)} jobs async")
        
        response_data = await self._make_request(
            method="POST",
            endpoint="v1/delete_jobs",
            data=request.model_dump(exclude_none=True),
        )
        
        return V1DeleteJobsResponse(**response_data)

    async def find_matching_jobs(
        self, request: V1FindMatchingJobsRequest
    ) -> V1FindMatchingJobsResponse:
        """
        Find jobs relevant to a candidate profile - async version

        Args:
            request: Job matching request with candidate profile

        Returns:
            Matching jobs response with relevance scores
        """
        logger.info(f"Finding matching jobs async for profile (limit={request.limit})")

        response_data = await self._make_request(
            method="POST",
            endpoint="v1/find_matching_jobs",
            data=request.model_dump(exclude_none=True),
        )

        return V1FindMatchingJobsResponse(**response_data)

    async def get_profile(self, request: V1ProfileRequest) -> V1ProfileResponse:
        """
        Retrieve a user profile by LinkedIn profile ID - async version

        Args:
            request: Profile request with docid and options

        Returns:
            Profile response with detailed candidate information
        """
        logger.info(f"Retrieving profile async for docid: {request.docid}")

        response_data = await self._make_request(
            method="GET",
            endpoint="v1/profile",
            params=request.model_dump(exclude_none=True),
        )
        return V1ProfileResponse(**response_data)

    async def api_call_history(self, request: V1ApiCallHistoryRequest) -> V1ApiCallHistoryResponse:
        """
        Retrieve API call history for the authenticated user - async version

        Get a list of previous API calls including their parameters,
        results, and metadata.

        Args:
            request: API call history request with limit and optional paths parameters

        Returns:
            API call history response with list of previous API calls and total credits used
        """
        logger.info(f"Retrieving API call history async with limit: {request.limit}")

        params = request.model_dump(exclude_none=True)
        if params.get("paths"):
            params["paths"] = ",".join(params["paths"])
        response_data = await self._make_request(
            method="GET",
            endpoint="v1/api_call_history",
            params=params,
        )

        return V1ApiCallHistoryResponse(**response_data)

    async def get_user(self) -> V1UserResponse:
        """
        Get user information, remaining credits, and pricing details - async version

        Retrieves information about the authenticated user including their
        email, API key details, remaining credits, and current pricing
        configuration for different operations.

        Returns:
            User response with user info, credits remaining, and pricing details
        """
        logger.info("Retrieving user information async")

        response_data = await self._make_request(
            method="GET",
            endpoint="v1/user"
        )

        return V1UserResponse(**response_data)
